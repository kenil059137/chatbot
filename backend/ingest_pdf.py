import os
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# Set Tesseract path (IMPORTANT)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def ingest_pdf():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    pdf_folder = os.path.join(BASE_DIR, "data", "pdf")

    CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

    documents = []

    for file in os.listdir(pdf_folder):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(pdf_folder, file)

            print("Processing:", pdf_path)

            images = convert_from_path(
                pdf_path,
                poppler_path=r"C:\cpp\Release-25.12.0-0\poppler-25.12.0\Library\bin"
            )

            for i, image in enumerate(images):

                text = pytesseract.image_to_string(image)

                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": file, "page": i}
                        )
                    )

    print("Extracted pages:", len(documents))

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print("Created chunks:", len(chunks))

    # Embedding
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Store
    db = Chroma.from_documents(
    chunks,
    embedding,
    persist_directory=CHROMA_PATH
    )

    db.persist()

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_pdf()
    
