import os
from groq import Groq
from modules import medisense_ml

_client = None

def _get_client():
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        print("[rag_chatbot] GROQ_API_KEY not set.")
        return None
    try:
        _client = Groq(api_key=api_key)
        return _client
    except Exception as e:
        print(f"[rag_chatbot] Failed to init Groq client: {e}")
        return None


def rag_answer(question, top_k=3):
    """
    Retrieves relevant Q&A and generates a natural answer using Llama-3.1-8B on Groq.
    Returns a dict: { 'answer': str, 'sources': list, 'used_llm': bool }
    """
    # 1. Retrieve relevant context using semantic search
    results = medisense_ml.semantic_search_qa(question, top_k=top_k)

    if not results:
        return {
            'answer': None,
            'sources': [],
            'used_llm': False,
        }

    # 2. Build the prompt with the retrieved context
    context_blocks = []
    sources = []
    for i, r in enumerate(results, 1):
        context_blocks.append(f"Source {i}: {r['question']}\n{r['answer']}")
        sources.append(r.get('question', 'Unknown'))

    context = "\n\n".join(context_blocks)

    prompt = f"""You are MediSense, a helpful health information assistant for South Africa.
Answer the user's question using ONLY the information provided in the sources below.
If the sources don't contain the answer, say so honestly and recommend seeing a doctor.
Be concise and friendly. Do not invent medical facts.

SOURCES:
{context}

USER QUESTION: {question}

ANSWER:"""

    # 3. Call the LLM
    client = _get_client()
    if client is None:
        return {
            'answer': None,
            'sources': sources,
            'used_llm': False,
        }

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-70b-versatile",
            temperature=0.3,
            max_tokens=500,
        )
        natural_answer = chat_completion.choices[0].message.content.strip()
        return {
            'answer': natural_answer,
            'sources': sources,
            'source_label': 'AI (Llama 3.1)',
            'used_llm': True,
        }
    except Exception as e:
        print(f"[rag_chatbot] Groq API call failed: {e}")
        return {
            'answer': None,
            'sources': sources,
            'used_llm': False,
        }