from langchain_core.runnables import RunnableLambda
from .retriever import retrieve_with_scores
from .generator import generate_answer
from .critic import verify_answer
from ..rag_chain import gemini_llm


def _rewrite(x):
    if not x.get("history", "").strip():
        return {**x, "standalone_question": x["question"]}
    try:
        prompt = f"""Rewrite the follow up question as a complete standalone question.
Keep ALL specific details from history.

Conversation History:
{x['history']}

Follow Up Question: {x['question']}

Output ONLY the rewritten question:"""

        rewritten = gemini_llm(prompt).strip()
        print(f"Original: {x['question']}")
        print(f"Rewritten: {rewritten}")
        return {**x, "standalone_question": rewritten}

    except Exception as e:
        print(f"Rewrite failed: {e}")
        return {**x, "standalone_question": x["question"]}


def _retrieve(x):
    try:
        docs, confidence, level = retrieve_with_scores(x["standalone_question"])
        context = "\n\n".join(d.page_content for d in docs)
        print(f"Context length: {len(context)}, Confidence: {level}")
        return {
            **x,
            "context": context,
            "confidence": confidence,
            "confidence_level": level
        }
    except Exception as e:
        print(f"Retrieval error: {e}")
        return {
            **x,
            "context": "",
            "confidence": 0.0,
            "confidence_level": "none"
        }


def _generate(x):
    answer = generate_answer(x["question"], x["context"], x.get("history", ""))

    if x["confidence_level"] == "low":
        is_valid, status = verify_answer(x["question"], x["context"], answer)
        print(f"Critic: {status}")
        if not is_valid:
            answer = "I'm not confident enough. Please visit www.charusat.ac.in"
            return {**x, "answer": answer, "confidence": 0.3, "confidence_level": "low"}
    else:
        print(f"Critic: SKIPPED (confidence: {x['confidence_level']})")

    return {**x, "answer": answer}


rag_chain = (
    RunnableLambda(_rewrite)
    | RunnableLambda(_retrieve)
    | RunnableLambda(_generate)
)


def multi_agent_rag(question: str, session_id=None, history: str = "") -> tuple:
    result = rag_chain.invoke({"question": question, "history": history})
    return result["answer"], result["confidence"], result["confidence_level"]