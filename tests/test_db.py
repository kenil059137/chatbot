# test_db.py — run from project root
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

count = db._collection.count()
print("Total documents in DB:", count)

if count > 0:
    results = db.similarity_search("student support policy", k=3)
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:300])

print(db._collection.count())