import os
import shutil
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import SeleniumURLLoader
from dotenv import load_dotenv

load_dotenv()

# paths from env or defaults
TESSERACT_PATH = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
POPPLER_PATH   = os.getenv("POPPLER_PATH",   r"C:\cpp\Release-25.12.0-0\poppler-25.12.0\Library\bin")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_FOLDER  = os.path.join(BASE_DIR, "data", "pdf")
URLS_FILE   = os.path.join(BASE_DIR, "data", "web", "urls.txt")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# ========================
# Category Mappings
# ========================
URL_CATEGORIES = {
    "course-single":  {"type": "web", "category": "courses"},
    "scholarship":    {"type": "web", "category": "scholarship"},
    "exam-corner":    {"type": "web", "category": "exam"},
    "admission-doc":  {"type": "web", "category": "admission"},
    "student-corner": {"type": "web", "category": "student_support"},
    "calendar":       {"type": "web", "category": "academic_calendar"},
}

PDF_CATEGORIES = {
    "student_support_policy": {"type": "pdf", "category": "financial_support"},
    "UGSF":                   {"type": "pdf", "category": "financial_support"},
    "PGSF":                   {"type": "pdf", "category": "financial_support"},
    "admission":              {"type": "pdf", "category": "admission"},
    "scholarship":            {"type": "pdf", "category": "scholarship"},
    "exam":                   {"type": "pdf", "category": "exam"},
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


def is_useful_content(text):
    return len(text.strip()) >= 500


def clean_content(text, category):
    # for course pages — remove navigation
    if category == "courses":
        lines = text.split("\n")
        cleaned = []
        start = False
        for line in lines:
            if any(k in line for k in ["Faculty:", "Institute:", "Duration:", "Eligibility", "Bachelor", "Master", "Programme"]):
                start = True
            if start and line.strip():
                cleaned.append(line.strip())
        result = "\n".join(cleaned)
        return result if len(result) > 200 else text  # fallback to full text
    # for other pages — return as is
    return text


# ========================
# 1. PDF Loader (OCR)
# ========================
def load_pdfs(documents):
    if not os.path.exists(PDF_FOLDER):
        print("No pdf folder found, skipping.")
        return

    for file in os.listdir(PDF_FOLDER):
        if not file.endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_FOLDER, file)
        print(f"\nProcessing PDF: {file}")
        meta = get_pdf_category(file)

        images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
        print(f"Total pages: {len(images)}")

        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            print(f"  Page {i+1} length: {len(text.strip())}")
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": file, "page": i, "type": meta["type"], "category": meta["category"]}
                ))


# ========================
# 2. Web Loader (Selenium)
# ========================
def load_web(documents):
    if not os.path.exists(URLS_FILE):
        print("No urls.txt found, skipping web.")
        return

    with open(URLS_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    if not urls:
        return

    print(f"\nScraping {len(urls)} URLs...")
    loader = SeleniumURLLoader(urls=urls)
    docs = loader.load()

    for doc in docs:
        url = doc.metadata['source']
        meta = get_url_category(url)
        content = clean_content(doc.page_content, meta["category"])

        print(f"Loaded: {url} — {len(content)} chars — category: {meta['category']}")

        if is_useful_content(content):
            documents.append(Document(
                page_content=content,
                metadata={"source": url, "type": meta["type"], "category": meta["category"]}
            ))
        else:
            print(f"  Skipped — too short after cleaning")

    web_count = len([d for d in documents if d.metadata['type'] == 'web'])
    print(f"Total web docs added: {web_count}")


# ========================
# 3. Ingest All
# ========================
def ingest_all():
    print("=== Starting Ingestion ===\n")

    documents = []  # local variable — safe!

    load_pdfs(documents)
    load_web(documents)

    print(f"\nTotal documents loaded: {len(documents)}")

    if not documents:
        print("ERROR: No documents found!")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"Total chunks: {len(chunks)}")

    # Category breakdown
    categories = {}
    for chunk in chunks:
        cat = chunk.metadata.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    print("\nChunks by category:")
    for cat, count in categories.items():
        print(f"  {cat}: {count} chunks")

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("\nOld ChromaDB cleared.")

    db = Chroma.from_documents(chunks, embedding, persist_directory=CHROMA_PATH)
    print(f"\nIngestion complete. Total chunks stored: {db._collection.count()}")


if __name__ == "__main__":
    ingest_all()