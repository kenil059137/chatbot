import os
import shutil
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import SeleniumURLLoader

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\cpp\Release-25.12.0-0\poppler-25.12.0\Library\bin"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_FOLDER  = os.path.join(BASE_DIR, "data", "pdf")
URLS_FILE   = os.path.join(BASE_DIR, "data", "web", "urls.txt")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

documents = []


# ========================
# URL → Category mapping
# ========================
URL_CATEGORIES = {
    "course-single":           {"type": "web", "category": "courses"},
    "scholarship":             {"type": "web", "category": "scholarship"},
    "exam-corner":             {"type": "web", "category": "exam"},
    "admission-doc":           {"type": "web", "category": "admission"},
    "student-corner":          {"type": "web", "category": "student_support"},
    "calendar":                {"type": "web", "category": "academic_calendar"},
}

# PDF filename → Category mapping
PDF_CATEGORIES = {
    "student_support_policy":  {"type": "pdf", "category": "financial_support"},
    "UGSF":                    {"type": "pdf", "category": "financial_support"},
    "PGSF":                    {"type": "pdf", "category": "financial_support"},
    "admission":               {"type": "pdf", "category": "admission"},
    "scholarship":             {"type": "pdf", "category": "scholarship"},
    "exam":                    {"type": "pdf", "category": "exam"},
}


def get_pdf_category(filename):
    for keyword, meta in PDF_CATEGORIES.items():
        if keyword.lower() in filename.lower():
            return meta
    return {"type": "pdf", "category": "general"}


def get_url_category(url):
    for keyword, meta in URL_CATEGORIES.items():
        if keyword in url:
            return meta
    return {"type": "web", "category": "general"}


# ========================
# 1. PDF Loader (OCR)
# ========================
def load_pdfs():
    if not os.path.exists(PDF_FOLDER):
        print("No pdf folder found, skipping.")
        return

    for file in os.listdir(PDF_FOLDER):
        if not file.endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_FOLDER, file)
        print(f"\nProcessing PDF: {file}")

        meta = get_pdf_category(file)

        images = convert_from_path(
            pdf_path,
            dpi=300,
            poppler_path=POPPLER_PATH
        )

        print(f"Total pages: {len(images)}")

        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            print(f"  Page {i+1} length: {len(text.strip())}")

            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source":   file,
                        "page":     i,
                        "type":     meta["type"],
                        "category": meta["category"]
                    }
                ))


# ========================
# 2. Web Loader (Selenium)
# ========================
def load_web():
    if not os.path.exists(URLS_FILE):
        print("No urls.txt found, skipping web.")
        return

    with open(URLS_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    if not urls:
        print("urls.txt is empty, skipping web.")
        return

    print(f"\nScraping {len(urls)} URLs...")

    loader = SeleniumURLLoader(urls=urls)
    docs = loader.load()

    for doc in docs:
        url = doc.metadata['source']
        meta = get_url_category(url)

        print(f"Loaded: {url} — {len(doc.page_content)} chars — category: {meta['category']}")

        # split by sections for better chunking
        sections = doc.page_content.split("\n\n")
        for section in sections:
            if len(section.strip()) > 150:
                documents.append(Document(
                    page_content=section.strip(),
                    metadata={
                        "source":   url,
                        "type":     meta["type"],
                        "category": meta["category"]
                    }
                ))

    print(f"Total web sections loaded: {len([d for d in documents if d.metadata['type'] == 'web'])}")


# ========================
# 3. Ingest All
# ========================
def ingest_all():
    print("=== Starting Ingestion ===\n")

    load_pdfs()
    load_web()

    print(f"\nTotal documents loaded: {len(documents)}")

    if not documents:
        print("ERROR: No documents found!")
        return

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks: {len(chunks)}")

    # Print category breakdown
    categories = {}
    for chunk in chunks:
        cat = chunk.metadata.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    print("\nChunks by category:")
    for cat, count in categories.items():
        print(f"  {cat}: {count} chunks")

    # Embedding
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Clear old DB
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("\nOld ChromaDB cleared.")

    # Store
    db = Chroma.from_documents(chunks, embedding, persist_directory=CHROMA_PATH)
    print(f"\nIngestion complete. Total chunks stored: {db._collection.count()}")


if __name__ == "__main__":
    ingest_all()





###########
###RAG CHAIN FOR ONLY BACKup#############

import os
import ssl
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

# SSL bypass for college WiFi — remove before production
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

from google import genai

# =========================
# Gemini Setup
# =========================
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def gemini_llm(prompt):
    if hasattr(prompt, "to_string"):
        prompt = prompt.to_string()

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )
    return response.text


# =========================
# Vector DB Retriever
# =========================
from backend.services.vector_store import vector_db

retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 30}
)


# =========================
# Prompt Template
# =========================
prompt = ChatPromptTemplate.from_template("""
You are a professional CHARUSAT university student support chatbot.

RULES:
- Answer ONLY using the context provided
- Detect language of question and reply in SAME language
- For listing questions, share what you have and say
  "For complete list visit www.charusat.ac.in"
- Never make up information not in context

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer clearly and professionally in the same language as the question:
""")


# =========================
# Format Docs
# =========================
def format_docs(docs):
    if not docs:
        return "No relevant information found."
    return "\n\n".join(doc.page_content for doc in docs)


# =========================
# RAG Chain
# =========================
rag_chain = (
    {
        "context": lambda x: format_docs(
            retriever.invoke(x if isinstance(x, str) else x["question"])
        ),
        "question": lambda x: x if isinstance(x, str) else x["question"],
        "history": lambda x: "" if isinstance(x, str) else x.get("history", ""),
    }
    | prompt
    | gemini_llm
)


if __name__ == "__main__":
    while True:
        question = input("Ask: ")
        result = rag_chain.invoke(question)
        print("Answer:", result)


#################################################################################################