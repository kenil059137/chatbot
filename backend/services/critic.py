from ..rag_chain import gemini_llm


def verify_answer(question, context, answer):

    prompt = f"""
Check if the answer is supported by the context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Respond with only:
VALID or INVALID
"""

    result = gemini_llm(prompt)

    return "VALID" in result.upper()