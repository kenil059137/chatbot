import re
from .vector_store import vector_db
from .confidence import calculate_confidence


def get_category_filter(query):
    query_lower = query.lower()

    # Use word boundaries to avoid false matches like "coffee", "referee"
    def has_word(word):
        return bool(re.search(rf'\b{re.escape(word)}\b', query_lower))

    if any(has_word(w) for w in ["course", "program", "b.tech", "bca", "mba", "intake", "duration"]):
        return "courses"
    if any(has_word(w) for w in ["scholarship", "financial aid", "stipend"]):
        return "scholarship"
    if any(has_word(w) for w in ["exam", "result", "examination", "sessional"]):
        return "exam"
    if any(has_word(w) for w in ["admit", "document", "apply", "application", "admission"]):
        return "admission"
    if any(has_word(w) for w in ["reimbursement", "conference", "research paper", "student support"]):
        return "financial_support"
    if any(has_word(w) for w in ["fee", "eligibility"]):
        return "courses"

    return None


def retrieve_with_scores(query):

    category = get_category_filter(query)

    if category:
        print(f"Filtering by category: {category}")
        docs_with_scores = vector_db.similarity_search_with_score(
            query,
            k=8,
            filter={"category": category}
        )
    else:
        docs_with_scores = []

    # Fallback to general search if category filter returns nothing
    if not docs_with_scores:
        if category:
            print("No category match — falling back to general search")
        docs_with_scores = vector_db.similarity_search_with_score(query, k=8)

    if not docs_with_scores:
        return [], 0.0, "none"

    docs = [doc for doc, _ in docs_with_scores]
    scores = [score for _, score in docs_with_scores]

    # Pass source info for traceability
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        print(f"  Retrieved: {source}")

    confidence, confidence_level = calculate_confidence(scores)

    return docs, confidence, confidence_level