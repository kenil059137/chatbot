from ..rag_chain import gemini_llm

MAX_HISTORY_CHARS = 1500
MAX_CONTEXT_CHARS = 12000


def truncate(text, max_chars):
    if len(text) <= max_chars:
        return text
    # For history — keep most recent (end)
    return text[-max_chars:]


def generate_answer(question, context, history=""):

    if not context.strip():
        context = "No relevant information found in university documents."

    history = truncate(history, MAX_HISTORY_CHARS)
    context = truncate(context, MAX_CONTEXT_CHARS)

    prompt = f"""You are a professional CHARUSAT university student support chatbot.

RULES:

- Answer using ONLY the context provided
- You are allowed to interpret and summarize relevant information even if exact wording is not present
- If the question is general (like "what is policy"), explain based on available details in context
- Do NOT say "not mentioned" if related information exists
- For personal questions (name, previous messages) — answer from CONVERSATION HISTORY
- If user says their name, remember it and use it
- Detect language and reply in same language
- Never make up information
- Only say "not available" if context is completely unrelated
  * For greetings like "hi/hello" — greet back warmly
  * For general questions — say you can help with university related queries
- Match detail level to what user asked
- List ALL relevant items found — never cut short
- Reply in same language as question
- Never make up information

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer clearly:"""

    return gemini_llm(prompt)