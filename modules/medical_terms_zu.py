# ============================================================
# MediSense — English ↔ isiZulu Medical Term Bridge
# ============================================================
# This module provides bidirectional translation between English
# and isiZulu medical terms. It is deliberately small and curated
# so that medical accuracy is preserved.
#
# Structure:
#   EN_TO_ZU  : dict mapping English term (lowercase) -> isiZulu
#   ZU_TO_EN  : auto-generated reverse lookup
#   ZU_QUESTION_WORDS : isiZulu question words for language detection
#   ZU_VOCAB : set of all isiZulu terms (used for detection)
# ============================================================

EN_TO_ZU = {
    # ---------- Common symptoms ----------
    "pain": "ubuhlungu",
    "ache": "ubuhlungu",
    "headache": "ikhanda elibuhlungu",
    "stomach ache": "isisu esibuhlungu",
    "back pain": "ubuhlungu bemhlane",
    "chest pain": "ubuhlungu besifuba",
    "sore throat": "umphimbo obuhlungu",
    "toothache": "izinyo elibuhlungu",
    "ear pain": "indlebe ebuhlungu",
    "joint pain": "ubuhlungu bamalunga",
    "muscle pain": "ubuhlungu bemisipha",

    "cough": "ukukhwehlela",
    "fever": "umkhuhlane (fever)",
    "chills": "ukuqhaqhazela",
    "sweating": "ukujuluka",
    "rash": "ukuqubuka",
    "itching": "ukulunywa",
    "swelling": "ukuvuvukala",
    "vomiting": "ukuhlanza",
    "nausea": "isicanucanu",
    "diarrhoea": "uhudo",
    "diarrhea": "uhudo",
    "constipation": "ukuqunjelwa",
    "fatigue": "ukukhathala",
    "weakness": "ubuthakathaka",
    "dizziness": "isiyezi",
    "fainting": "ukuquleka",
    "bleeding": "ukopha",
    "runny nose": "ukuhlanza amakhala",
    "sneezing": "ukuthimula",
    "shortness of breath": "ukuphefumula kanzima",
    "difficulty breathing": "ukuphefumula kanzima",
    "chest tightness": "ukucinana kwesifuba",
    "wheezing": "ukukhwehlela okunomsindo",
    "blurred vision": "ukubona kancane",
    "loss of appetite": "ukungadli",
    "weight loss": "ukwehla kwesisindo",
    "weight gain": "ukwenyuka kwesisindo",
    "thirst": "ukoma",
    "frequent urination": "ukuchama kaningi",
    "painful urination": "ukuchama kubuhlungu",
    "insomnia": "ukungaphumuli",
    "anxiety": "ukukhathazeka",
    "depression": "ukudangala",
    "confusion": "ukudideka",
    "memory loss": "ukukhohlwa",
    "seizure": "isithuthwane",
    "paralysis": "ukukhubazeka",
    "numbness": "ukubanda emzimbeni",

    # ---------- Common diseases ----------
    "flu": "umkhuhlane",
    "influenza": "umkhuhlane",
    "cold": "umkhuhlane",
    "common cold": "umkhuhlane",
    "covid": "covid",
    "covid-19": "covid-19",
    "malaria": "malaria",
    "tuberculosis": "isifo sofuba",
    "tb": "isifo sofuba",
    "hiv": "i-hiv",
    "aids": "i-aids",
    "diabetes": "isifo sikashukela",
    "high blood pressure": "umfutho wegazi ophezulu",
    "hypertension": "umfutho wegazi ophezulu",
    "low blood pressure": "umfutho wegazi ophansi",
    "asthma": "isifuba somoya",
    "pneumonia": "inyumoniya",
    "bronchitis": "isifo somphimbo",
    "cancer": "umdlavuza",
    "ulcer": "isilonda",
    "stomach ulcer": "isilonda esiswini",
    "peptic ulcer": "isilonda esiswini",
    "heart attack": "ukuhlaselwa yinhliziyo",
    "stroke": "ukushaywa umgogodla",
    "epilepsy": "isithuthwane",
    "meningitis": "isifo sobuchopho",
    "measles": "isimungumungwane",
    "chickenpox": "ingqambo",
    "hepatitis": "isifo sesibindi",
    "jaundice": "i-jaundice",
    "anaemia": "ukushoda kwengqondo",
    "anemia": "ukushoda kwengqondo",
    "arthritis": "amathambo",
    "gout": "i-gout",
    "allergy": "ukungezwani komzimbo",
    "allergic reaction": "ukungezwani komzimbo",
    "food poisoning": "ubuthi bokudla",
    "diarrhoeal disease": "isifo sohudo",
    "typhoid": "i-typhoid",
    "cholera": "i-kholera",
    "dengue": "i-dengue",
    "rabies": "i-rabies",
    "tetanus": "i-tetanus",
    "shingles": "i-shingles",
    "acne": "amabhamuza",
    "eczema": "i-eczema",
    "psoriasis": "i-psoriasis",
    "fungal infection": "isifo sokhunta",
    "ringworm": "ikhwezane",
    "athlete's foot": "isifo sonyawo",
    "yeast infection": "isifo sekhambi",
    "conjunctivitis": "isifo samehlo",
    "pink eye": "isifo samehlo",
    "migraine": "ikhanda elibuhlungu kakhulu",
    "appendicitis": "isifo se-appendix",
    "hernia": "i-hernia",
    "kidney stones": "amatshe ezinso",
    "urinary tract infection": "isifo somchamo",
    "uti": "isifo somchamo",
    "sexually transmitted infection": "isifo socansi",
    "sti": "isifo socansi",
    "std": "isifo socansi",
    "gonorrhoea": "i-gonorrhoea",
    "syphilis": "i-syphilis",
    "chlamydia": "i-chlamydia",
    "herpes": "i-herpes",
    "dementia": "ukuwohloka komqondo",
    "alzheimer's": "i-alzheimer's",
    "parkinson's": "i-parkinson's",

    # ---------- First aid / emergency ----------
    "emergency": "isimo esiphuthumayo",
    "ambulance": "i-ambulensi",
    "hospital": "isibhedlela",
    "clinic": "umtholampilo",
    "doctor": "udokotela",
    "nurse": "umhlengikazi",
    "medicine": "umuthi",
    "medicines": "imithi",
    "treatment": "ukwelashwa",
    "diagnosis": "ukuxilongwa",
    "prevention": "ukuvimbela",
    "vaccine": "umuthi wokugoma",
    "vaccination": "ukugoma",
    "surgery": "ukuhlinzwa",
    "bandage": "ibhandeshi",
    "wound": "inxeba",
    "cut": "ukusikeka",
    "burn": "ukusha",
    "fracture": "ukuphuka",
    "broken bone": "ithambo eliphukile",
    "sprain": "ukuklwebheka",
    "bite": "ukulunywa",
    "snake bite": "ukulunywa yinyoka",
    "dog bite": "ukulunywa yinja",
    "insect bite": "ukulunywa isinambuzane",
    "poisoning": "ubuthi",
    "overdose": "ukweqisa umuthi",
    "cpr": "i-cpr",
    "choking": "ukuklinywa",
    "drowning": "ukuminza",
    "unconscious": "ukuquleka",
    "bleeding heavily": "ukopha kakhulu",

    # ---------- Body parts ----------
    "head": "ikhanda",
    "eye": "ihlo",
    "eyes": "amehlo",
    "ear": "indlebe",
    "nose": "ikhala",
    "mouth": "umlomo",
    "throat": "umphimbo",
    "neck": "intamo",
    "chest": "isifuba",
    "heart": "inhliziyo",
    "lungs": "amaphaphu",
    "stomach": "isisu",
    "liver": "isibindi",
    "kidney": "inso",
    "kidneys": "izinso",
    "bladder": "isinye",
    "blood": "igazi",
    "bone": "ithambo",
    "bones": "amathambo",
    "muscle": "umsipha",
    "skin": "isikhumba",
    "brain": "ubuchopho",
    "arm": "ingalo",
    "leg": "umlenze",
    "hand": "isandla",
    "foot": "unyawo",
    "back": "umhlane",
    "abdomen": "isisu",
    "joint": "ilunga",
    "joints": "amalunga",
    "vein": "umthambo",
    "nerve": "inzwa",

    # ---------- Question phrases ----------
    "what is": "yini",
    "what are": "yini",
    "how do i": "ngenzani",
    "how can i": "ngingenzani",
    "why": "kungani",
    "when": "nini",
    "where": "kuphi",
    "who": "ubani",
    "which": "yiphi",
    "can i": "ngingakwazi",
    "should i": "kufanele",
    "help me": "ngisize",

    # ---------- Simple actions ----------
    "yes": "yebo",
    "no": "cha",
    "thank you": "ngiyabonga",
    "please": "ngicela",
    "sorry": "ngiyaxolisa",
    "i am sick": "ngiyagula",
    "i have": "ngina",
    "i feel": "ngizwa",
    "i need": "ngidinga",
    "give me": "nginike",
    "tell me": "ngitshele",
    "explain": "chaza",
}


# Auto-generated reverse dictionary
# Note: when multiple English terms map to the same isiZulu, the FIRST one wins.
ZU_TO_EN = {}
for _en, _zu in EN_TO_ZU.items():
    _zu_lower = _zu.lower().strip()
    if _zu_lower and _zu_lower not in ZU_TO_EN:
        ZU_TO_EN[_zu_lower] = _en


# isiZulu question words for language detection
ZU_QUESTION_WORDS = {
    "yini", "kungani", "kanjani", "ngenzani", "ngingenzani",
    "nini", "kuphi", "ubani", "yiphi", "ngubani", "kufanele",
    "ngingakwazi", "ngisize", "ngitshele", "chaza", "ngicela",
}


# Full isiZulu vocabulary set — used for language detection
ZU_VOCAB = set()
for _zu in EN_TO_ZU.values():
    for _word in _zu.lower().replace("-", " ").split():
        if len(_word) > 1:
            ZU_VOCAB.add(_word)
ZU_VOCAB.update(ZU_QUESTION_WORDS)


# Common isiZulu prefixes (for detection when terms are conjugated)
ZU_PREFIXES = (
    "ngi", "nga", "ngu", "ngi", "wa", "ba", "ya", "yi", "ku", "ka",
    "zi", "li", "lu", "si", "bu", "mu",
)