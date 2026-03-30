import os
import re
import time
import shutil
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import SeleniumURLLoader
from dotenv import load_dotenv

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

load_dotenv()

TESSERACT_PATH = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
POPPLER_PATH   = os.getenv("POPPLER_PATH",   r"C:\cpp\Release-25.12.0-0\poppler-25.12.0\Library\bin")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_FOLDER  = os.path.join(BASE_DIR, "data", "pdf")
URLS_FILE   = os.path.join(BASE_DIR, "data", "web", "urls.txt")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# ── Category mappings ──────────────────────────────────────────────────────────

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

LARGE_CHUNK_CATEGORIES = {"financial_support", "scholarship"}

SKIP_KEYWORDS = [
    "info@charusat", "+91 2697", "Mon - Sat", "How to Reach",
    "Quick Links", "Exam Result", "Pay Fees", "Downloads",
    "Alumni Portal", "Donations", "Visit Website", "Enquire Now",
    "Cookie", "Privacy Policy", "All Rights Reserved", "Follow Us",
    "Announcements", "University Announcements", "Contact",
    "CHARUSAT Campus", "Off. Nadiad", "Changa-388421"
]

OCR_NOISE_PATTERNS = [
    r'\b[A-Z0-9]{1,2}\b(?:\s+[A-Z0-9]{1,2}\b){4,}',
    r'[|]{2,}',
    r'_{3,}',
    r'\.{4,}',
    r'\s{3,}',
]


# ── Category helpers ───────────────────────────────────────────────────────────

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


# ── Content cleaning ───────────────────────────────────────────────────────────

def clean_web_content(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(skip in line for skip in SKIP_KEYWORDS):
            continue
        cleaned.append(line)
    result = "\n".join(cleaned)
    return result if len(result) > 200 else text


def clean_ocr_text(text):
    for pattern in OCR_NOISE_PATTERNS:
        if r'\s{3,}' in pattern:
            text = re.sub(pattern, ' ', text)
        else:
            text = re.sub(pattern, '', text)

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            cleaned.append("")
            continue
        if re.fullmatch(r'[\d\s]+', line):
            continue
        if len(line) < 40 and line == line.upper() and not any(c.isdigit() for c in line):
            continue
        if any(skip in line for skip in SKIP_KEYWORDS):
            continue
        cleaned.append(line)

    result = re.sub(r'\n{3,}', '\n\n', "\n".join(cleaned))
    return result.strip()


def is_useful_content(text, min_chars=300):
    return len(text.strip()) >= min_chars


# ── JS-rendered course page scraper ───────────────────────────────────────────

def scrape_course_page(url):
    """
    /course-single is Next.js client-rendered. body.text gives flat plain
    text with no headings, so regex splitting on #### never works and
    the fallback splitter cuts mid-course, producing orphaned fee/eligibility
    chunks with no course name attached.

    Fix: extract course blocks directly from the DOM.
    Each course on the page is an <h4> followed by a <table>. We walk
    every <h4>, grab its text as the course name, then use JavaScript to
    find the next sibling <table> and read its innerText. This gives one
    clean string per course that always starts with the course name —
    no post-processing splitting needed at all.

    Returns a list of per-course strings (one per programme).
    """
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    driver = webdriver.Chrome(options=options)
    course_blocks = []

    try:
        print(f"    Loading JS-rendered page: {url}")
        driver.get(url)

        # Wait until course headings appear — confirms JS fetch is done
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "h4"))
        )
        time.sleep(2)  # buffer for lazy-loaded lower sections

        headings = driver.find_elements(By.CSS_SELECTOR, "h4")
        print(f"    Found {len(headings)} course headings in DOM")

        for h4 in headings:
            course_name = h4.text.strip()
            if not course_name or len(course_name) < 5:
                continue

            # DOM structure: grandparent > [div > h4] + [div.meta-post > table]
            # h4.nextElementSibling is null — table is in a sibling of h4's PARENT.
            # Go up one level first, then walk parent's siblings to find the table.
            table_text = driver.execute_script("""
                var h = arguments[0];
                var parentDiv = h.parentElement;
                if (!parentDiv) return null;
                var sib = parentDiv.nextElementSibling;
                while (sib) {
                    var t = sib.querySelector('table');
                    if (t) return t.innerText;
                    
                    sib = sib.nextElementSibling;
                }
                return null;
            """, h4)

            block_lines = [course_name]
            if table_text:
                for line in table_text.split("\n"):
                    line = line.strip()
                    if line:
                        block_lines.append(line)
            else:
                parent_text = driver.execute_script("""
    var el = arguments[0].closest('section');
    if (el) return el.innerText;
    return arguments[0].parentElement.innerText;
""", h4)
                block_lines.append(parent_text)

            block = "\n".join(block_lines)

            # Keep blocks that have real detail beyond just the course name
            if len(block) > 80:
                course_blocks.append(block)

        print(f"    Extracted {len(course_blocks)} clean course blocks from DOM")

    except Exception as e:
        print(f"    scrape_course_page failed: {e}")
    finally:
        driver.quit()

    return course_blocks


# ── Splitter selector ──────────────────────────────────────────────────────────

def get_splitter(category):
    if category in LARGE_CHUNK_CATEGORIES:
        return RecursiveCharacterTextSplitter(
            chunk_size=1800,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )


# ── PDF loading ────────────────────────────────────────────────────────────────

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

        ocr_config = r"--oem 3 --psm 6 -l eng"

        try:
            images = convert_from_path(pdf_path, dpi=400, poppler_path=POPPLER_PATH)
        except Exception as e:
            print(f"  Could not convert {file}: {e}")
            continue

        print(f"  Total pages: {len(images)}")

        for i, image in enumerate(images):
            raw_text = pytesseract.image_to_string(image, config=ocr_config)
            cleaned  = clean_ocr_text(raw_text)

            print(f"  Page {i+1}: {len(raw_text.strip())} raw → {len(cleaned.strip())} cleaned chars")

            if is_useful_content(cleaned, min_chars=200):
                documents.append(Document(
                    page_content=cleaned,
                    metadata={
                        "source":   file,
                        "page":     i + 1,
                        "type":     meta["type"],
                        "category": meta["category"],
                    }
                ))
            else:
                print(f"  Page {i+1} skipped — too little usable content after cleaning")


# ── Web loading ────────────────────────────────────────────────────────────────

def load_web(documents):
    if not os.path.exists(URLS_FILE):
        print("No urls.txt found, skipping web.")
        return

    with open(URLS_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    if not urls:
        return

    print(f"\nScraping {len(urls)} URLs one by one...")

    for url in urls:
        try:
            print(f"  Scraping: {url}")
            meta = get_url_category(url)

            # course-single is Next.js client-rendered — DOM extraction needed
            if "course-single" in url:
                course_blocks = scrape_course_page(url)

                if not course_blocks:
                    print(f"    No course blocks extracted — JS may not have loaded")
                    continue

                for block in course_blocks:
                    documents.append(Document(
                        page_content=block,
                        metadata={
                            "source":   url,
                            "type":     meta["type"],
                            "category": meta["category"],
                        }
                    ))
                print(f"    Added {len(course_blocks)} course documents")
                continue

            # All other URLs — SeleniumURLLoader works fine
            loader = SeleniumURLLoader(
                urls=[url],
                arguments=[
                    "--headless",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--page-load-strategy=eager",
                ]
            )
            docs = loader.load()

            for doc in docs:
                src     = doc.metadata.get("source", url)
                content = clean_web_content(doc.page_content)

                print(f"    {src} — {len(content)} chars — [{meta['category']}]")

                if is_useful_content(content):
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source":   src,
                            "type":     meta["type"],
                            "category": meta["category"],
                        }
                    ))
                else:
                    print(f"    Skipped — too short after cleaning")

        except Exception as e:
            print(f"  Failed {url}: {e}")
            continue

    web_count = len([d for d in documents if d.metadata.get("type") == "web"])
    print(f"\nTotal web docs added: {web_count}")


# ── Main ingestion ─────────────────────────────────────────────────────────────

def ingest_all():
    print("=== Starting Ingestion ===\n")

    documents = []
    load_pdfs(documents)
    load_web(documents)

    print(f"\nTotal raw documents loaded: {len(documents)}")

    if not documents:
        print("ERROR: No documents found!")
        return

    # Course documents from scrape_course_page are already one-per-course —
    # no further splitting needed. All other categories use the standard splitter.
    all_chunks = []
    for doc in documents:
        category = doc.metadata.get("category", "general")
        source   = doc.metadata.get("source", "")

        if category == "courses" and "course-single" in source:
            # Already perfectly chunked — one doc = one course entry
            all_chunks.append(doc)
        else:
            splitter = get_splitter(category)
            chunks   = splitter.split_documents([doc])
            all_chunks.extend(chunks)

    print(f"Total chunks after splitting: {len(all_chunks)}")

    category_counts = {}
    for chunk in all_chunks:
        cat = chunk.metadata.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1
    print("\nChunks by category:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count} chunks")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("\nOld ChromaDB cleared.")

    db = Chroma.from_documents(
        all_chunks,
        embedding,
        persist_directory=CHROMA_PATH,
        collection_name="charusat_docs"
    )
    print(f"\nIngestion complete. Total chunks stored: {db._collection.count()}")


if __name__ == "__main__":
    ingest_all()