from google import genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Put your Gemini API key
client = genai.Client(api_key="AIzaSyBPq3NfVwzn44eOpkaYZtap2B1WVTsuJSg")

CHROMA_PATH = "chroma_db"

# Load embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector DB
vector_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding
)

retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)



def ask_chatbot(question):

    docs = retriever.invoke(question)

    print("\nDEBUG: Retrieved docs:", len(docs))

    context = "\n\n".join([doc.page_content for doc in docs])

    print("\nDEBUG: Context:", context[:300])

    prompt = f"""
You are a university chatbot assistant.

Use the context below to answer the question.

If context exists, answer clearly and completely.

If the question is general, explain based on context.

Context:
{context}

Question:
{question}

Answer:
"""



    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    return response.text



if __name__ == "__main__":

    while True:

        question = input("\nAsk question: ")

        if question == "exit":
            break

        answer = ask_chatbot(question)

        print("\nAnswer:", answer)
