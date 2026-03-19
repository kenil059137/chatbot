from .vector_store import vector_db
from .confidence import calculate_confidence


def get_category_filter(query):
    query_lower = query.lower()

    if any(w in query_lower for w in ["course", "program", "b.tech", "bca", "mba", "intake", "duration"]):
        return "courses"
    if any(w in query_lower for w in ["scholarship", "financial aid", "stipend"]):
        return "scholarship"
    if any(w in query_lower for w in ["exam", "result", "examination", "sessional"]):
        return "exam"
    if any(w in query_lower for w in ["admit", "document", "apply", "application"]):
        return "admission"
    if any(w in query_lower for w in ["reimbursement", "conference", "research paper"]):
        return "financial_support"
    if any(w in query_lower for w in ["fee", "eligibility"]):
        return "courses"
    return None


def retrieve_with_scores(query):

    category = get_category_filter(query)
    docs = []

    if category:
        print(f"Filtering by category: {category}")
        docs = vector_db.max_marginal_relevance_search(
            query,
            k=8,
            fetch_k=20,
            filter={"category": category}
        )

    # fallback to general search if category filter returns nothing
    if not docs:
        print("No category match or empty result — using general search")
        docs = vector_db.max_marginal_relevance_search(
            query,
            k=8,
            fetch_k=20
        )

    # similarity search for confidence scores
    docs_with_scores = vector_db.similarity_search_with_score(query, k=8)

    if not docs_with_scores:
        return docs, 0

    scores = [score for _, score in docs_with_scores]
    confidence = calculate_confidence(scores)

    return docs, confidence