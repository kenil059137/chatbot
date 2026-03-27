from langchain_core.runnables import RunnableLambda
from .retriever import retrieve_with_scores
from .generator import generate_answer
from .critic import verify_answer
from ..rag_chain import gemini_llm


def _rewrite(x):
    if not x.get("history", "").strip():
        return {**x, "standalone_question": x["question"]}
    try:
        rewritten = gemini_llm(
            f"Rewrite as standalone question.\n"
            f"History: {x['history']}\n"
            f"Follow up: {x['question']}\n"
            f"Rewritten:"
        ).strip()
        return {**x, "standalone_question": rewritten}
    except:
        return {**x, "standalone_question": x["question"]}


def _retrieve(x):
    docs, confidence, level = retrieve_with_scores(x["standalone_question"])
    return {
        **x,
        "context": "\n\n".join(d.page_content for d in docs),
        "confidence": confidence,
        "confidence_level": level
    }


def _generate(x):
    if not x["context"].strip():
        return {**x, "answer": "I couldn't find relevant info. Visit www.charusat.ac.in", "confidence": 0.0, "confidence_level": "none"}
    answer = generate_answer(x["question"], x["context"], x.get("history", ""))
    is_valid, status = verify_answer(x["question"], x["context"], answer)
    print(f"Critic: {status}")
    if not is_valid:
        answer = "I'm not confident enough. Visit www.charusat.ac.in"
        return {**x, "answer": answer, "confidence": min(x["confidence"], 0.3)}
    return {**x, "answer": answer}


rag_chain = (
    RunnableLambda(_rewrite)
    | RunnableLambda(_retrieve)
    | RunnableLambda(_generate)
)


def multi_agent_rag(question: str, session_id=None, history: str = "") -> tuple:
    result = rag_chain.invoke({"question": question, "history": history})
    return result["answer"], result["confidence"], result["confidence_level"]