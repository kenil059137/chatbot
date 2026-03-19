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