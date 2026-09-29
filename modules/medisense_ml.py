# ============================================================
# MediSense — Machine Learning Module
# ============================================================
# Central place for all ML / NLP logic:
#   - Semantic search (sentence-transformers)
#   - Symptom extraction from free text
#   - Spell correction (SymSpell)
#   - Clinic ranking
#   - Language detection (English / isiZulu)
#   - Bilingual term translation
#   - Health score (weighted nonlinear)
#
# Every function is wrapped so that if a model fails to load,
# it falls back to keyword-based behaviour instead of crashing.
# ============================================================

import os
import re
import math
import threading

# ------------------------------------------------------------------
# LAZY IMPORTS + GLOBALS
# ------------------------------------------------------------------
# We do not import sentence_transformers or symspellpy at module
# top-level because they are heavy and slow (they take several
# seconds to import). They are loaded once, on first use.
# ------------------------------------------------------------------

_ST_MODEL = None
_ST_LOCK = threading.Lock()
_ST_LOAD_FAILED = False

_SS_DICT = None
_SS_LOCK = threading.Lock()
_SS_LOAD_FAILED = False

# In-memory embedding caches — populated once
_QA_EMBEDDINGS = None
_QA_ROWS = None
_SYMPTOM_EMBEDDINGS = None
_SYMPTOM_ROWS = None

# Term bridge (English <-> isiZulu)
try:
    from modules.medical_terms_zu import (
        EN_TO_ZU, ZU_TO_EN, ZU_VOCAB, ZU_QUESTION_WORDS
    )
    _ZU_LOADED = True
except Exception as _e:
    print(f"[medisense_ml] could not load medical_terms_zu: {_e}")
    EN_TO_ZU = {}
    ZU_TO_EN = {}
    ZU_VOCAB = set()
    ZU_QUESTION_WORDS = set()
    _ZU_LOADED = False


# ==================================================================
# 1. SENTENCE-TRANSFORMER MODEL LOADER
# ==================================================================
# Loads all-MiniLM-L6-v2 once, thread-safe, cached forever.
# ==================================================================

def _get_st_model():
    """Return the SentenceTransformer model, or None if unavailable."""
    global _ST_MODEL, _ST_LOAD_FAILED
    if _ST_MODEL is not None:
        return _ST_MODEL
    if _ST_LOAD_FAILED:
        return None
    with _ST_LOCK:
        if _ST_MODEL is not None:
            return _ST_MODEL
        if _ST_LOAD_FAILED:
            return None
        try:
            print("[medisense_ml] loading sentence-transformers model...")
            from sentence_transformers import SentenceTransformer
            _ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            print("[medisense_ml] sentence-transformers model loaded")
            return _ST_MODEL
        except Exception as e:
            print(f"[medisense_ml] ST load failed: {e}")
            _ST_LOAD_FAILED = True
            return None


def _embed(texts):
    """Return a list of 384-dim vectors for the given texts, or None."""
    model = _get_st_model()
    if model is None:
        return None
    try:
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    except Exception as e:
        print(f"[medisense_ml] embed failed: {e}")
        return None


def _cosine_sim_matrix(query_vec, matrix):
    """Cosine similarity between one query vector and a matrix of vectors."""
    import numpy as np
    q = np.asarray(query_vec, dtype=float)
    m = np.asarray(matrix, dtype=float)
    q_norm = q / (np.linalg.norm(q) + 1e-10)
    m_norm = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-10)
    return m_norm @ q_norm


# ==================================================================
# 2. SYMSPELL LOADER (spell correction)
# ==================================================================

def _get_symspell(max_edit_distance=2):
    """
    Build a SymSpell dictionary from all the medical vocabulary we
    know about: symptoms, diseases, medicines, clinics, and common
    English words found in the Q&A dataset.

    Returns the SymSpell object or None if the library is missing.
    """
    global _SS_DICT, _SS_LOAD_FAILED
    if _SS_DICT is not None:
        return _SS_DICT
    if _SS_LOAD_FAILED:
        return None
    with _SS_LOCK:
        if _SS_DICT is not None:
            return _SS_DICT
        if _SS_LOAD_FAILED:
            return None
        try:
            from symspellpy import SymSpell
            ss = SymSpell(max_dictionary_edit_distance=max_edit_distance,
                          prefix_length=7)
            # NOTE: The actual vocabulary is injected later, from
            # routes.py, via build_spell_dictionary(terms). This is
            # because the vocabulary depends on the loaded DataFrames
            # which live in routes.py.
            _SS_DICT = ss
            return ss
        except Exception as e:
            print(f"[medisense_ml] SymSpell load failed: {e}")
            _SS_LOAD_FAILED = True
            return None


def build_spell_dictionary(terms):
    """
    Given a list of strings, add them all to the SymSpell dictionary.
    Call this once at startup from routes.py, after CSVs are loaded.

    Example:
        build_spell_dictionary(['diabetes', 'flu', 'headache', ...])
    """
    ss = _get_symspell()
    if ss is None:
        return False
    try:
        added = 0
        for term in terms:
            if not term:
                continue
            term = str(term).lower().strip()
            if len(term) < 3:
                continue
            # SymSpell accepts "word count" pairs; use count=1
            try:
                ss.create_dictionary_entry(term, 1)
                added += 1
            except Exception:
                pass
        print(f"[medisense_ml] SymSpell dictionary built with {added} terms")
        return True
    except Exception as e:
        print(f"[medisense_ml] failed to build spell dictionary: {e}")
        return False


# ==================================================================
# 3. LANGUAGE DETECTION (English vs isiZulu)
# ==================================================================

def detect_language(text):
    """
    Return 'zu' if the text looks like isiZulu, otherwise 'en'.
    Uses a simple heuristic based on the curated medical vocabulary.
    """
    if not text:
        return 'en'
    text_lower = text.lower()
    tokens = re.findall(r"[a-z']+", text_lower)
    if not tokens:
        return 'en'

    zu_hits = 0
    for tok in tokens:
        if tok in ZU_VOCAB:
            zu_hits += 1
        elif tok in ZU_QUESTION_WORDS:
            zu_hits += 1
        elif len(tok) > 4 and any(tok.startswith(p) for p in
                                  ('ngi', 'nga', 'ngu', 'uku', 'isi', 'ama',
                                   'aba', 'umu', 'aba')):
            zu_hits += 0.5  # weak hint

    # Consider it isiZulu if a meaningful share of tokens are isiZulu
    ratio = zu_hits / max(len(tokens), 1)
    return 'zu' if ratio >= 0.30 else 'en'


# ==================================================================
# 4. BILINGUAL TERM TRANSLATION
# ==================================================================

def translate_term(text, direction='en->zu'):
    """
    Translate a short phrase term-by-term.
    direction='en->zu' or 'zu->en'.

    This is a best-effort translation. Unknown words pass through
    unchanged so no information is lost.
    """
    if not text:
        return text
    src_dict = EN_TO_ZU if direction == 'en->zu' else ZU_TO_EN

    # First try full-phrase match
    key = text.strip().lower()
    if key in src_dict:
        return src_dict[key]

    # Then translate word-by-word
    tokens = re.findall(r"[A-Za-z']+", text)
    out_tokens = []
    for tok in tokens:
        t_low = tok.lower()
        if t_low in src_dict:
            out_tokens.append(src_dict[t_low])
        else:
            out_tokens.append(tok)
    if not out_tokens:
        return text
    return ' '.join(out_tokens)


# ==================================================================
# 5. SYMPTOM EXTRACTION (semantic)
# ==================================================================

_SYMPTOM_STOPWORDS = {
    'i', 'me', 'my', 'have', 'has', 'had', 'the', 'a', 'an', 'and',
    'or', 'but', 'with', 'of', 'to', 'for', 'in', 'on', 'at',
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'feel', 'feeling', 'felt', 'get', 'getting', 'got',
    'pain', 'ache',
}


def _tokenize_query(text):
    """Return a clean list of significant query tokens."""
    if not text:
        return []
    tokens = re.findall(r"[a-z']+", text.lower())
    return [t for t in tokens if t not in _SYMPTOM_STOPWORDS and len(t) > 2]


def extract_symptoms(text, symptom_vocab=None, top_k=5):
    """
    Given a free-text query, return a list of symptom terms
    present in the text.

    Strategy:
      1. Direct substring match against symptom_vocab
      2. Semantic similarity via sentence embeddings (if available)
      3. Fallback: keyword match

    Returns:
        List of dicts: [{'symptom': ..., 'score': 0.0-1.0}, ...]
    """
    if not text:
        return []

    text_lower = text.lower()
    results = []

    # --- Step 1: substring / exact match ---
    if symptom_vocab:
        for sym in symptom_vocab:
            if not sym:
                continue
            s_low = str(sym).lower().strip()
            if len(s_low) < 3:
                continue
            # Match if the symptom appears as a phrase in the text
            if s_low in text_lower:
                results.append({'symptom': sym, 'score': 1.0})

    # If we already found matches, we are done
    if results:
        # Dedupe and cap
        seen = set()
        unique = []
        for r in results:
            key = r['symptom'].lower()
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique[:top_k]

    # --- Step 2: semantic embedding match ---
    model = _get_st_model()
    if model is not None and symptom_vocab:
        try:
            import numpy as np
            voc = [str(s).strip() for s in symptom_vocab if s and len(str(s)) > 2]
            if not voc:
                return []
            vecs = _embed(voc)
            q_vec = _embed([text])
            if vecs is not None and q_vec is not None:
                sims = _cosine_sim_matrix(q_vec[0], vecs)
                idx = np.argsort(-sims)[:top_k]
                for i in idx:
                    score = float(sims[i])
                    if score < 0.30:
                        continue
                    results.append({'symptom': voc[i], 'score': round(score, 3)})
                return results
        except Exception as e:
            print(f"[medisense_ml] extract_symptoms semantic failed: {e}")

    # --- Step 3: fallback keyword match ---
    tokens = _tokenize_query(text)
    if symptom_vocab and tokens:
        for sym in symptom_vocab:
            s_low = str(sym).lower().strip()
            for tok in tokens:
                if len(tok) > 3 and tok in s_low:
                    results.append({'symptom': sym, 'score': 0.5})
                    break

    # Dedupe
    seen = set()
    unique = []
    for r in results:
        key = r['symptom'].lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique[:top_k]


# ==================================================================
# 6. SEMANTIC Q&A SEARCH (chatbot)
# ==================================================================

def _build_qa_index(qa_rows):
    """
    Precompute embeddings for a list of Q&A rows.
    Each row is a dict with at least 'question' and 'answer'.
    Called once per process.
    """
    global _QA_EMBEDDINGS, _QA_ROWS
    if _QA_EMBEDDINGS is not None and _QA_ROWS is not None:
        return True
    if not qa_rows:
        return False

    # Build the text we will embed: "question + tags"
    texts = []
    for row in qa_rows:
        q = str(row.get('question', '')).strip()
        tags = str(row.get('tags', '')).strip()
        if tags:
            texts.append(f"{q} {tags}")
        else:
            texts.append(q)

    vecs = _embed(texts)
    if vecs is None:
        return False

    _QA_EMBEDDINGS = vecs
    _QA_ROWS = qa_rows
    print(f"[medisense_ml] Q&A index built: {len(qa_rows)} rows")
    return True


def semantic_search_qa(query, top_k=3):
    """
    Search the pre-built Q&A index for the best matches to `query`.

    Returns:
        List of dicts: [{'question', 'answer', 'score', 'source'}, ...]
        Empty list if the index is not ready or the query is empty.
    """
    import numpy as np
    if not query or _QA_EMBEDDINGS is None or _QA_ROWS is None:
        return []

    q_vec = _embed([query])
    if q_vec is None:
        return []

    sims = _cosine_sim_matrix(q_vec[0], _QA_EMBEDDINGS)
    idx = np.argsort(-sims)[:top_k]

    results = []
    for i in idx:
        score = float(sims[i])
        if score < 0.20:
            continue
        row = _QA_ROWS[i]
        results.append({
            'question': row.get('question', ''),
            'answer': row.get('answer', ''),
            'source': row.get('source', row.get('tags', 'MediSense Guide')),
            'score': round(score, 3),
        })
    return results


# ==================================================================
# 7. SPELL CORRECTION (SymSpell)
# ==================================================================

def correct_spelling_symspell(query, max_suggestions=1):
    """
    Return the best spelling suggestion for `query`, or None.
    Uses SymSpell if the dictionary is ready, else returns None
    (routes.py will fall back to difflib).
    """
    if not query or len(query.strip()) < 3:
        return None
    ss = _get_symspell()
    if ss is None or getattr(ss, 'word_count', 0) == 0:
        return None

    q_lower = query.strip().lower()
    try:
        suggestions = ss.lookup_compound(q_lower, max_edit_distance=2)
        if suggestions and suggestions[0].term != q_lower:
            return suggestions[0].term
    except Exception as e:
        print(f"[medisense_ml] symspell lookup failed: {e}")
    return None


# ==================================================================
# 8. CLINIC RANKING
# ==================================================================

def _tokenize_place(text):
    """Tokenise a location string for matching."""
    if not text:
        return set()
    return set(re.findall(r"[a-z]+", text.lower()))


def rank_clinics(clinics, location, limit=15):
    """
    Rank a list of clinic dicts by how well they match `location`.

    Scoring factors:
      - Exact match on City/Area/Clinic_Name/Province  -> 100
      - Substring match                                -> 60
      - Token overlap                                  -> up to 40
      - Clinic type priority (Hospital > CHC > Clinic) -> 0-5

    Returns a new list of clinic dicts, best match first, each with
    an added 'match_score' field.
    """
    if not clinics:
        return []

    loc_tokens = _tokenize_place(location)

    def _clinic_score(c):
        # The clinic dict is from pandas .to_dict('records'), so keys
        # are the column names. We are defensive about missing keys.
        name = str(c.get('Clinic_Name', '')).lower()
        city = str(c.get('City', '')).lower()
        area = str(c.get('Area', '')).lower()
        prov = str(c.get('Province', '')).lower()
        ctype = str(c.get('Clinic_Type', '')).lower()

        score = 0.0

        if not location:
            # No filter — just use clinic type priority
            pass
        else:
            loc_low = location.lower().strip()
            for field in (city, area, name, prov):
                if not field:
                    continue
                if loc_low == field:
                    score = max(score, 100)
                elif loc_low in field or field in loc_low:
                    score = max(score, 60)

            # Token overlap
            field_tokens = _tokenize_place(f"{name} {city} {area} {prov}")
            if loc_tokens and field_tokens:
                overlap = len(loc_tokens & field_tokens)
                if overlap:
                    score = max(score, 20 + overlap * 10)

        # Clinic type priority
        if 'hospital' in ctype:
            score += 5
        elif 'chc' in ctype or 'community health' in ctype:
            score += 3
        elif 'clinic' in ctype:
            score += 1

        return score

    scored = []
    for c in clinics:
        c_copy = dict(c)
        c_copy['match_score'] = _clinic_score(c)
        scored.append(c_copy)

    # Sort highest first, then by name for stable ordering
    scored.sort(key=lambda x: (-x['match_score'], str(x.get('Clinic_Name', ''))))
    return scored[:limit]


# ==================================================================
# 9. HEALTH SCORE (weighted nonlinear)
# ==================================================================

def compute_health_score(
    age=30,
    has_id=False,
    has_phone=False,
    has_gender=False,
    has_emergency=False,
    has_allergies=False,
    has_health_conditions=False,
    appointments=None,
):
    """
    Return an integer health score in [0, 100].

    This is NOT machine-learned — it is a weighted nonlinear function
    designed to be transparent and tunable. Every point is justified.

    Weights:
      Profile completeness      ~35 pts (nonlinear — each field is worth less)
      Appointment engagement    ~40 pts (logarithmic — first 3 matter most)
      Recency                   ~15 pts (if active in last 30 days)
      Data quality              ~10 pts
      Penalties (no-shows, cancels)  up to -30 pts
    """
    from datetime import datetime, timedelta

    appointments = appointments or []
    score = 0.0

    # --- Profile completeness (nonlinear) ---
    # Ordering: id > gender > phone > emergency > allergies > health conditions
    weights = [
        (has_id, 10),
        (has_gender, 6),
        (has_phone, 6),
        (has_emergency, 5),
        (has_allergies, 4),
        (has_health_conditions, 4),
    ]
    for present, w in weights:
        if present:
            score += w

    # --- Age sanity bonus ---
    if age and 1 <= age <= 120 and age != 30:
        score += 5

    # --- Appointment engagement (logarithmic) ---
    completed = sum(1 for a in appointments if a['status'] == 'Completed')
    checked_in = sum(1 for a in appointments if a['status'] == 'Checked-in')
    total = len(appointments)

    # log scale: 1 appt = 5, 3 appt = 12, 5 appt = 17, 10 appt = 26
    engagement = math.log10(completed + 1) * 17
    engagement += math.log10(checked_in + 1) * 8
    score += min(engagement, 40)

    # --- Recency bonus ---
    if appointments:
        try:
            most_recent = max(
                datetime.strptime(a['appointment_date'], '%Y-%m-%d')
                for a in appointments
                if a['appointment_date']
            )
            days_ago = (datetime.now() - most_recent).days
            if days_ago <= 30:
                score += 15
            elif days_ago <= 90:
                score += 8
            elif days_ago <= 180:
                score += 3
        except Exception:
            pass

    # --- Penalties ---
    no_shows = sum(1 for a in appointments if a['status'] == 'No-Show')
    cancelled = sum(1 for a in appointments if a['status'] == 'Cancelled')
    score -= min(no_shows * 8, 25)
    score -= min(cancelled * 2, 8)

    # --- Regular booking bonus ---
    if total >= 3: score += 3
    if total >= 5: score += 3
    if total >= 10: score += 3

    # Clamp
    return int(max(0, min(score, 100)))


def health_score_category(score):
    """Return a human-readable category for a health score."""
    if score >= 80: return 'Excellent'
    if score >= 60: return 'Good'
    if score >= 40: return 'Fair'
    if score >= 20: return 'Needs Attention'
    return 'New'


# ==================================================================
# 10. BILINGUAL QUERY NORMALIZATION
# ==================================================================

_NOISE_WORDS = {
    'i', 'a', 'an', 'the', 'is', 'are', 'was', 'were', 'of', 'to', 'in',
    'on', 'at', 'by', 'for', 'with', 'and', 'or', 'but', 'do', 'does',
    'did', 'it', 'this', 'that',
    'what', 'how', 'why', 'when', 'where', 'who', 'whom', 'whose',
}


def normalize_query_for_search(query):
    """
    Convert a user query (English or isiZulu) to a clean English query.
    """
    if not query:
        return query, 'en'
    lang = detect_language(query)
    if lang == 'en':
        return query, 'en'

    translated = translate_term(query, direction='zu->en')

    # Remove noise words that break semantic search
    tokens = translated.split()
    cleaned = [t for t in tokens if t.lower() not in _NOISE_WORDS]
    # If everything was noise, keep the original translation
    cleaned_query = ' '.join(cleaned) if cleaned else translated

    return cleaned_query, 'zu'


# ==================================================================
# 11. BILINGUAL ANSWER TRANSLATION
# ==================================================================

def translate_answer_to_zulu(answer):
    """
    Return the answer unchanged.

    We do NOT translate the full answer word-by-word because the
    dictionary is too small (250 terms) — a word-by-word translation
    of a medical paragraph produces mixed English/isiZulu garbage.

    Query translation still works (so isiZulu searches find the right
    answer), but the answer itself is shown in English.

    To enable real isiZulu answers later, install a translation model
    (e.g. Helsinki-NLP/opus-mt-en-zu) and implement it here.
    """
    return answer


# ==================================================================
# 12. PUBLIC SETUP FUNCTION (called once from routes.py)
# ==================================================================

def initialize_ml(df_symptom_descriptions=None,
                  df_disease_master=None,
                  df_chatbot_qa=None,
                  df_medical_qa=None,
                  df_medicines=None,
                  df_clinics=None):
    """
    Prepare all ML subsystems at startup:
      - Load the sentence-transformer model
      - Build the spell dictionary from all known medical terms
      - Build the Q&A embedding index

    Call this ONCE, after DataFrames are loaded in routes.py.
    """
    print("[medisense_ml] initializing...")

    # --- Build symptom vocabulary ---
    symptom_vocab = set()
    try:
        if df_symptom_descriptions is not None and 'symptom' in df_symptom_descriptions.columns:
            for s in df_symptom_descriptions['symptom'].dropna().astype(str):
                if s.strip():
                    symptom_vocab.add(s.strip())
        if df_disease_master is not None and 'all_symptoms' in df_disease_master.columns:
            # all_symptoms is a list-like string; pull individual symptoms
            for cell in df_disease_master['all_symptoms'].dropna().astype(str):
                try:
                    items = eval(cell)
                    if isinstance(items, (list, tuple)):
                        for s in items:
                            if s and isinstance(s, str) and len(s.strip()) > 2:
                                symptom_vocab.add(s.strip())
                except Exception:
                    pass
    except Exception as e:
        print(f"[medisense_ml] symptom vocab build error: {e}")
    print(f"[medisense_ml] symptom vocabulary: {len(symptom_vocab)} items")

    # --- Build spell dictionary ---
    spell_terms = set()
    for df, col in [
        (df_symptom_descriptions, 'symptom'),
        (df_disease_master, 'disease'),
        (df_chatbot_qa, 'tags'),
        (df_medical_qa, 'question'),
        (df_medicines, 'medicine'),
        (df_clinics, 'Clinic_Name'),
        (df_clinics, 'City'),
        (df_clinics, 'Area'),
        (df_clinics, 'Province'),
    ]:
        try:
            if df is not None and col in df.columns:
                for val in df[col].dropna().astype(str):
                    val = val.strip().lower()
                    if 3 <= len(val) <= 40 and ' ' not in val:
                        spell_terms.add(val)
        except Exception:
            pass
    # Add all medical terms from the zu dictionary (english side)
    for en_term in EN_TO_ZU.keys():
        for word in en_term.split():
            if len(word) > 2:
                spell_terms.add(word)
    build_spell_dictionary(list(spell_terms))

    # --- Load the ST model (warms up the cache) ---
    _get_st_model()

    # --- Build Q&A index (merge chatbot_qa first, then medical_qa) ---
    qa_rows = []
    try:
        if df_chatbot_qa is not None and len(df_chatbot_qa) > 0:
            for _, row in df_chatbot_qa.iterrows():
                qa_rows.append({
                    'question': str(row.get('question', '')),
                    'answer': str(row.get('answer', '')),
                    'tags': str(row.get('tags', '')),
                    'source': 'MediSense Guide',
                })
        if df_medical_qa is not None and len(df_medical_qa) > 0:
            for _, row in df_medical_qa.iterrows():
                qa_rows.append({
                    'question': str(row.get('question', '')),
                    'answer': str(row.get('answer', '')),
                    'tags': '',
                    'source': str(row.get('source', 'Medical Q&A')),
                })
    except Exception as e:
        print(f"[medisense_ml] Q&A row build error: {e}")

    if qa_rows:
        _build_qa_index(qa_rows)

    print("[medisense_ml] initialization complete")


# ==================================================================
# 13. HIGH-LEVEL BILINGUAL SEARCH (one-shot)
# ==================================================================

def bilingual_search(query, top_k=3):
    """
    End-to-end bilingual search:
      1. Detect query language
      2. Normalize to English if isiZulu
      3. Run semantic Q&A search
      4. If user asked in isiZulu, translate the top answer back

    Returns a dict:
        {
          'query': original,
          'language': 'en' or 'zu',
          'normalized_query': english version,
          'results': [ {question, answer, source, score}, ... ],
          'translated': bool,     # whether answer was translated
          'top_answer': str       # the (possibly translated) answer
        }
    """
    if not query:
        return {
            'query': '', 'language': 'en', 'normalized_query': '',
            'results': [], 'translated': False, 'top_answer': ''
        }

    lang = detect_language(query)
    normalized, _ = normalize_query_for_search(query)

    # Run semantic search on English-normalized query
    results = semantic_search_qa(normalized, top_k=top_k)

    top_answer = ''
    translated = False
    if results:
        top_answer = results[0].get('answer', '')

    return {
        'query': query,
        'language': lang,
        'normalized_query': normalized,
        'results': results,
        'translated': translated,
        'top_answer': top_answer,
    }


# ==================================================================
# 14. CONFIDENCE LABELS
# ==================================================================

def confidence_label(score):
    """
    Return a human-readable confidence label for a cosine similarity
    score in [0, 1].
    """
    if score is None:
        return 'Unknown'
    if score >= 0.75:
        return 'High'
    if score >= 0.55:
        return 'Medium'
    if score >= 0.35:
        return 'Low'
    return 'Very Low'


# ==================================================================
# 15. SYMPTOM EXTRACTION WRAPPER (public)
# ==================================================================
# routes.py calls this. It uses the module-level symptom vocabulary
# which is populated by initialize_ml().
# ==================================================================

_SYMPTOM_VOCAB_CACHE = None

def _get_symptom_vocab():
    """Return the cached symptom vocabulary (or empty list)."""
    global _SYMPTOM_VOCAB_CACHE
    if _SYMPTOM_VOCAB_CACHE is not None:
        return _SYMPTOM_VOCAB_CACHE
    _SYMPTOM_VOCAB_CACHE = []
    return _SYMPTOM_VOCAB_CACHE


def set_symptom_vocab(vocab_list):
    """Called by routes.py after loading CSVs, to register vocabulary."""
    global _SYMPTOM_VOCAB_CACHE
    _SYMPTOM_VOCAB_CACHE = list(vocab_list)


def extract_symptoms_public(text, top_k=5):
    """
    Public symptom extractor. Uses the registered symptom vocabulary.
    Returns a list of dicts: [{'symptom': ..., 'score': ...}, ...]
    """
    vocab = _get_symptom_vocab()
    return extract_symptoms(text, symptom_vocab=vocab, top_k=top_k)


# ==================================================================
# 16. STATUS / DIAGNOSTIC
# ==================================================================

def ml_status():
    """Return a dict summarising what is loaded. Useful for logging."""
    return {
        'sentence_transformers': _ST_MODEL is not None,
        'st_load_failed': _ST_LOAD_FAILED,
        'symspell': _SS_DICT is not None,
        'symspell_terms': getattr(_SS_DICT, 'word_count', 0) if _SS_DICT else 0,
        'qa_index_size': len(_QA_ROWS) if _QA_ROWS else 0,
        'symptom_vocab_size': len(_get_symptom_vocab()),
        'zulu_bridge': _ZU_LOADED,
    }


# ==================================================================
# 17. EXPORTS
# ==================================================================

__all__ = [
    # Setup
    'initialize_ml',
    'build_spell_dictionary',
    'set_symptom_vocab',

    # Language
    'detect_language',
    'translate_term',
    'translate_answer_to_zulu',
    'normalize_query_for_search',

    # Symptom extraction
    'extract_symptoms',
    'extract_symptoms_public',

    # Chatbot / Q&A
    'semantic_search_qa',
    'bilingual_search',
    'confidence_label',

    # Spell correction
    'correct_spelling_symspell',

    # Clinics
    'rank_clinics',

    # Health score
    'compute_health_score',
    'health_score_category',

    # Diagnostics
    'ml_status',
]