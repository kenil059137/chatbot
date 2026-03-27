from ..rag_chain import gemini_llm

MAX_HISTORY_CHARS = 1500
MAX_CONTEXT_CHARS = 3000


def truncate(text, max_chars):
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]  # keep most recent content


def generate_answer(question, context, history=""):

    if not context.strip():
        context = "No relevant information found."

    # Truncate to avoid exceeding Gemini context limits
    history = truncate(history, MAX_HISTORY_CHARS)
    context = truncate(context, MAX_CONTEXT_CHARS)

    prompt = f"""You are a professional CHARUSAT university student support chatbot.

RULES:
- Answer ONLY using the context below
- If context is insufficient, politely say so and suggest visiting www.charusat.ac.in
- If question is in Gujarati reply in Gujarati
- If question is in Hindi reply in Hindi  
- If question is in English reply in English
- For listing questions share what you have and say "For complete list visit www.charusat.ac.in"
- Never make up information not in context

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer clearly and professionally in the same language as the question:"""

    return gemini_llm(prompt)