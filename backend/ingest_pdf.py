"""
Run this to add a single PDF to existing ChromaDB without re-scraping URLs.
Usage: python -m backend.ingest_single_pdf
"""
import os
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

TESSERACT_PATH = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
POPPLER_PATH   = os.getenv("POPPLER_PATH",   r"C:\cpp\Release-25.12.0-0\poppler-25.12.0\Library\bin")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# ── Config — change this to your PDF filename ──────────────────────────────────
PDF_FILENAME = "Government-Scholarship-2025-26.pdf"
CATEGORY     = "scholarship"
# ──────────────────────────────────────────────────────────────────────────────

PDF_PATH = os.path.join(BASE_DIR, "data", "pdf", PDF_FILENAME)


def ingest_single_pdf():
    if not os.path.exists(PDF_PATH):
        print(f"ERROR: File not found: {PDF_PATH}")
        return

    print(f"Processing: {PDF_FILENAME}")
    ocr_config = r"--oem 3 --psm 6 -l eng"

    images = convert_from_path(PDF_PATH, dpi=400, poppler_path=POPPLER_PATH)
    print(f"Total pages: {len(images)}")

    documents = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image, config=ocr_config).strip()
        print(f"  Page {i+1}: {len(text)} chars")
        if len(text) > 100:
            documents.append(Document(
                page_content=text,
                metadata={
                    "source":   PDF_FILENAME,
                    "page":     i + 1,
                    "type":     "pdf",
                    "category": CATEGORY,
                }
            ))

    if not documents:
        print("ERROR: No text extracted!")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks: {len(chunks)}")

    # Load embedding and ADD to existing ChromaDB — no clearing
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    db = Chroma(
        collection_name="charusat_docs",
        persist_directory=CHROMA_PATH,
        embedding_function=embedding
    )

    db.add_documents(chunks)
    print(f"Done! Added {len(chunks)} chunks to existing ChromaDB.")
    print(f"Total chunks now: {db._collection.count()}")


if __name__ == "__main__":
    ingest_single_pdf()