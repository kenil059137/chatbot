from .retriever import retrieve_with_scores
from .generator import generate_answer
from ..rag_chain import gemini_llm


def rewrite_question(question, history):
    if not history or not history.strip():
        return question
    try:
        prompt = f"""Given the conversation history and a follow up question,
rewrite the follow up question as a complete standalone question.

Conversation History:
{history}

Follow Up Question: {question}

Rewritten Standalone Question (just the question, nothing else):"""

        rewritten = gemini_llm(prompt)
        print(f"Original: {question}")
        print(f"Rewritten: {rewritten}")
        return rewritten.strip()

    except Exception as e:
        print(f"Rewrite failed: {e}, using original question")
        return question


def multi_agent_rag(question, session_id=None, history=""):

    # Rewrite question using history for better retrieval
    standalone_question = rewrite_question(question, history)

    # Agent 1: Retriever
    docs, confidence = retrieve_with_scores(standalone_question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Agent 2: Generator
    answer = generate_answer(question, context, history)

    return answer, confidence


if __name__ == "__main__":
    while True:
        q = input("Ask: ")
        answer, conf = multi_agent_rag(q)
        print("\nAnswer:", answer)
        print("Confidence:", conf)
        print("\n-----------------\n")