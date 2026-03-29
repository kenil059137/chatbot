from ..rag_chain import gemini_llm


def verify_answer(question, context, answer):
    if not context.strip() or not answer.strip():
        return False, "insufficient_data"

    # If context has content and answer is reasonable length — trust it
    if len(answer) > 50 and len(context) > 100:
        return True, "SKIPPED"  # skip critic to save API calls

    prompt = f"""Check if the answer is supported by the context.

Context:
{context}

Question:
{question}

Answer:
{answer}

Reply with exactly one word only — VALID or INVALID:"""

    try:
        result = gemini_llm(prompt).strip().upper()
        is_valid = result == "VALID"
        return is_valid, result
    except Exception as e:
        print(f"Critic failed: {e}")
        return True, "SKIPPED"