import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load once at module level — no repeated downloads
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # better than MiniLM
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

vector_db = Chroma(
    collection_name="charusat_docs",  # explicit name — avoids default collection conflicts
    persist_directory=CHROMA_PATH,
    embedding_function=embedding
)