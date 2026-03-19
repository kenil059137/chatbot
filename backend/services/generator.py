from ..rag_chain import gemini_llm


def generate_answer(question, context, history=""):

    if not context.strip():
        context = "No relevant information found."

    prompt = f"""
You are a professional CHARUSAT university student support chatbot.

RULES:
- Answer ONLY using the context below
- If context is insufficient, politely say so and suggest visiting www.charusat.ac.in
- Detect language of question and reply in SAME language
- If question is in Gujarati reply in Gujarati
- If question is in Hindi reply in Hindi
- If question is in English reply in English
- For listing questions share what you have and say
  "For complete list visit www.charusat.ac.in"
- Never make up information not in context

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer clearly and professionally in the same language as the question:
"""
    return gemini_llm(prompt)