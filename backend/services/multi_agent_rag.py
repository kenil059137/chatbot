from .retriever import retrieve_with_scores
from .generator import generate_answer
from .critic import verify_answer


def multi_agent_rag(question, session_id=None):

    # Agent 1: Retriever
    docs, confidence = retrieve_with_scores(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    # Agent 2: Generator
    answer = generate_answer(question, context)

    # Agent 3: Critic
    is_valid = verify_answer(question, context, answer)

    if not is_valid:
        answer = "I could not verify a reliable answer. Please contact university support directly."

    return answer, confidence


if __name__ == "__main__":
    while True:
        q = input("Ask: ")
        answer, conf = multi_agent_rag(q)
        print("\nAnswer:", answer)
        print("Confidence:", conf)
        print("\n-----------------\n")