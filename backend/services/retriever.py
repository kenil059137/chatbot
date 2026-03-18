from .vector_store import vector_db
from .confidence import calculate_confidence


def retrieve_with_scores(query):

    # MMR retrieval (for diverse context)
    docs = vector_db.max_marginal_relevance_search(
        query,
        k=5,
        fetch_k=20
    )

    # Similarity search (for confidence scores)
    docs_with_scores = vector_db.similarity_search_with_score(query, k=5)

    if not docs_with_scores:
        return docs, 0

    scores = [score for _, score in docs_with_scores]
    confidence = calculate_confidence(scores)

    return docs, confidence