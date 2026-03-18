from ..rag_chain import gemini_llm


def generate_answer(question, context):

    prompt = f"""
You are a university student support chatbot.

Answer ONLY using the context below.

Context:
{context}

Question:
{question}

Answer clearly:
"""

    return gemini_llm(prompt)