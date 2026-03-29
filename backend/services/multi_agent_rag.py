from langchain_core.runnables import RunnableLambda
from .retriever import retrieve_with_scores
from .generator import generate_answer
from .critic import verify_answer
from ..rag_chain import gemini_llm


def _rewrite(x):
    if not x.get("history", "").strip():
        return {**x, "standalone_question": x["question"]}
    try:
        prompt = f"""You are a question rewriter. Given conversation history and a follow up question, rewrite the follow up as a complete standalone question that includes all necessary context from the history.

Conversation History:
{x['history']}

Follow Up Question: {x['question']}

Rules:
- Include specific subject names from history (e.g. "B.Tech courses" not just "these courses")
- Keep it as a single clear question
- Output ONLY the rewritten question, nothing else

Rewritten Question:"""

        rewritten = gemini_llm(prompt).strip()
        print(f"Original: {x['question']}")
        print(f"Rewritten: {rewritten}")
        return {**x, "standalone_question": rewritten}
    except:
        return {**x, "standalone_question": x["question"]}


def _retrieve(x):
    docs, confidence, level = retrieve_with_scores(x["standalone_question"])
    context = "\n\n".join(d.page_content for d in docs)
    print(f"Context length: {len(context)}")  # add this
    print(f"Confidence: {level}")              # add this
    print(f"Context preview: {context[:200]}") # add this
    return {
        **x,
        "context": context,
        "confidence": confidence,
        "confidence_level": level
    }


def _generate(x):
    if not x["context"].strip():
        return {
            **x,
            "answer": "I couldn't find relevant info. Visit www.charusat.ac.in",
            "confidence": 0.0,
            "confidence_level": "none"
        }

    answer = generate_answer(x["question"], x["context"], x.get("history", ""))

    # Only run critic when confidence is low — saves API calls
    if x["confidence_level"] == "low":
        is_valid, status = verify_answer(x["question"], x["context"], answer)
        print(f"Critic: {status}")
        if not is_valid:
            answer = "I'm not confident enough. Visit www.charusat.ac.in"
            return {**x, "answer": answer, "confidence": min(x["confidence"], 0.3)}
    else:
        print(f"Critic: SKIPPED (confidence is {x['confidence_level']})")

    return {**x, "answer": answer}


rag_chain = (
    RunnableLambda(_rewrite)
    | RunnableLambda(_retrieve)
    | RunnableLambda(_generate)
)


def multi_agent_rag(question: str, session_id=None, history: str = "") -> tuple:
    result = rag_chain.invoke({"question": question, "history": history})
    return result["answer"], result["confidence"], result["confidence_level"]