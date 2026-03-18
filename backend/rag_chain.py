import os

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory


from langchain_core.chat_history import InMemoryChatMessageHistory

from google import genai

#FOR COLLEGE ONLY 
import ssl
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context
#stilll here 

# =========================
# Gemini Setup
# =========================


load_dotenv()

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
# Embeddings + Vector DB
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding
)

retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)


# =========================
# Prompt Template (Advanced)
# =========================

prompt = ChatPromptTemplate.from_template("""
You are a professional university student support chatbot.

Use the conversation history and context to answer.

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer clearly and professionally:
""")


# =========================
# Format Retrieved Docs
# =========================

def format_docs(docs):
    if not docs:
        return "No relevant information found."
    return "\n\n".join(doc.page_content for doc in docs)


# =========================
# Base RAG Chain
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


# =========================
# SESSION MEMORY
# =========================

store = {}


def get_session_history(session_id: str):

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


# =========================
# FINAL ADVANCED CHAIN
# =========================

chat_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

if __name__ == "__main__":
    while True:
        question = input("Ask: ")
        result = rag_chain.invoke(question)
        print("Answer:", result)



# # ==============================
# # Gemini setup
# # ==============================

# client = genai.Client(api_key="AIzaSyBPq3NfVwzn44eOpkaYZtap2B1WVTsuJSg")


# def gemini_llm(prompt):

#     response = client.models.generate_content(
#         model="gemini-2.5-flash-lite",
#         contents=prompt
#     )

#     return response.text


# llm = RunnableLambda(gemini_llm)

# # ==============================
# # Load vector DB
# # ==============================

# embedding = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# vector_db = Chroma(
#     persist_directory=CHROMA_PATH,
#     embedding_function=embedding
# )


# retriever = vector_db.as_retriever(
#     search_type="mmr",
#     search_kwargs={"k": 5, "fetch_k": 20}
# )


# # ==============================
# # Prompt template
# # ==============================

# prompt_template = PromptTemplate.from_template("""
# You are an AI chatbot for university student support.

# Answer the question using ONLY the context provided.

# If context is insufficient, say politely that information is not available.
                                               
# Use conversation history and context to answer.

# Context:
# {context}

# Question:
# {question}

# Answer clearly and completely:
# """)


# # ==============================
# # Format docs function
# # ==============================

# def format_docs(docs):
#     if not docs:
#         return "No relevant information found."
#     return "\n\n".join(doc.page_content for doc in docs)


# # ==============================
# # Build chain
# # ==============================

# rag_chain = (
#     {
#         "context": retriever | RunnableLambda(format_docs),
#         "question": RunnablePassthrough(),
#         "history": RunnableLambda(lambda x: "")
#     }
#     | prompt_template
#     | llm
# )




# # ==============================
# # Chat loop
# # ==============================

# if __name__ == "__main__":

#     print("Student Support Chatbot Ready\n")

#     while True:

#         question = input("Ask question: ")

#         if question.lower() == "exit":
#             break

#         answer = rag_chain.invoke(question)

#         print("\nAnswer:", answer)
#         print("\n" + "-"*50 + "\n")


# # =========================
# # Memory store
# # =========================


# store = {}

# def get_session_history(session_id: str):
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]

# # Export chain so other files can use it
# __all__ = ["rag_chain"]