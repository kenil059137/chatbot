import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Absolute path fix
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# Load vector database
vector_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding
)