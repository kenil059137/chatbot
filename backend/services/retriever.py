from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from services.confidence import calculate_confidence


# Load embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load vector database
vector_db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)


def retrieve_with_scores(query):

    # Step 1 — MMR retrieval (for context)
    docs = vector_db.max_marginal_relevance_search(
        query,
        k=5,
        fetch_k=20
    )

    # Step 2 — similarity search (for scores)
    docs_with_scores = vector_db.similarity_search_with_score(query, k=5)

    print("Similarity results:", docs_with_scores)

    if not docs_with_scores:
        return docs, 0

    scores = [score for _, score in docs_with_scores]

    print("Distances:", scores)

    avg_distance = sum(scores) / len(scores)

    confidence = round(1 / (1 + avg_distance), 2)

    return docs, confidence