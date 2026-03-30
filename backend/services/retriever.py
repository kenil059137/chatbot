import re
from .vector_store import vector_db
from .confidence import calculate_confidence
from langchain_core.documents import Document

def get_category_filter(query):
    query_lower = query.lower()

    def has_word(word):
        return bool(re.search(rf'\b{re.escape(word)}\b', query_lower))

    if any(has_word(w) for w in ["course", "program", "b.tech", "bca", "mba", "intake", "duration", "offer", "available", "provide"]):
        return "courses"
    if any(has_word(w) for w in ["scholarship", "financial aid", "stipend"]):
        return "scholarship"
    if any(has_word(w) for w in ["exam", "result", "examination", "sessional"]):
        return "exam"
    if any(has_word(w) for w in ["admit", "document", "apply", "application", "admission"]):
        return "admission"
    if any(has_word(w) for w in ["reimbursement", "conference", "research paper", "student support", "policy"]):
        return "financial_support"
    if any(has_word(w) for w in ["fee", "eligibility"]):
        return "courses"
    return None


def is_listing_question(query):
    listing_words = ["what courses", "which courses", "list courses",
                     "what programs", "courses offered", "courses available",
                     "courses provide", "what btech", "what mba", "what bca",
                     "courses does", "courses do", "offer"]
    query_lower = query.lower()
    return any(w in query_lower for w in listing_words)


def get_all_category_docs(category):
    """Fetch ALL docs from a category directly from ChromaDB"""
    result = vector_db.get(where={"category": category})
    docs = []
    for i, content in enumerate(result["documents"]):
        meta = result["metadatas"][i]
        docs.append(Document(
            page_content=content,
            metadata=meta
        ))
    print(f"Fetched {len(docs)} docs from category: {category}")
    return docs


def retrieve_with_scores(query):

    # Skip retrieval for greetings
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "namaste"]
    if query.strip().lower() in greetings:
        return [], 1.0, "high"

    category = get_category_filter(query)

    # For listing questions — fetch ALL docs from category
    if category and is_listing_question(query):
       print(f"★ Listing question — fetching ALL {category} docs")
       docs = get_all_category_docs(category)
       print(f"★ Got {len(docs)} docs")
       return docs, 0.9, "high"

    # For specific questions — use MMR
    k = 10 if category else 8

    if category:
        print(f"Filtering by category: {category}")
        docs = vector_db.max_marginal_relevance_search(
            query, k=k, fetch_k=50,
            filter={"category": category}
        )
    else:
        docs = vector_db.max_marginal_relevance_search(
            query, k=k, fetch_k=30
        )

    if not docs:
        print("Empty — falling back to general MMR")
        docs = vector_db.max_marginal_relevance_search(query, k=8, fetch_k=30)

    if not docs:
        return [], 0.0, "none"

    # Filter junk
    junk_keywords = ["info@charusat", "Quick Links", "+91 2697",
                     "How to Reach", "Mon - Sat", "Pay Fees", "Downloads"]
    clean = [d for d in docs
             if len(d.page_content.strip()) > 200
             and not any(j in d.page_content for j in junk_keywords)]
    docs = clean if clean else docs

    for doc in docs:
        print(f"  Retrieved: {doc.metadata.get('source')} — {len(doc.page_content)} chars")

    # Confidence scores
    docs_with_scores = vector_db.similarity_search_with_score(query, k=5)
    scores = [s for _, s in docs_with_scores] if docs_with_scores else [0.5]
    confidence, level = calculate_confidence(scores)

    return docs, confidence, level