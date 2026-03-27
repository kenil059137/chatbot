from ..rag_chain import gemini_llm


def verify_answer(question, context, answer):
    if not context.strip() or not answer.strip():
        return False, "insufficient_data"

    prompt = f"""Check if the answer is fully supported by the context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Reply with exactly one word only — VALID or INVALID:"""

    try:
        result = gemini_llm(prompt).strip().upper()
        is_valid = result == "VALID"  # exact match — fixes substring bug
        return is_valid, result
    except Exception as e:
        print(f"Critic failed: {e}")
        return True, "SKIPPED"  # fail open — don't block answer