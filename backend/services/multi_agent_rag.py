from .retriever import retrieve_with_scores
from ..rag_chain import gemini_llm


def multi_agent_rag(question):

    # Agent 1: Retriever
    docs, confidence = retrieve_with_scores(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    # Agent 2: Generator
    prompt = f"""
You are an assistant answering based only on the provided context.

Context:
{context}

Question:
{question}

Answer clearly.
"""

    answer = gemini_llm(prompt)

    # Agent 3: Simple critic
    if not context.strip():
        answer = "I could not find relevant information in the documents."

    return answer, confidence


if __name__ == "__main__":

    while True:
        q = input("Ask: ")

        answer, conf = multi_agent_rag(q)

        print("\nAnswer:", answer)
        print("Confidence:", conf)
        print("\n-----------------\n")
print("Total docs:", vector_db._collection.count())