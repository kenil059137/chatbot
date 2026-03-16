from fastapi import FastAPI
from pydantic import BaseModel

from backend.rag_chain import chat_chain
from backend.services.retriever import retrieve_with_scores

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Advanced Student Support Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    session_id: str


@app.post("/chat")
def chat(request: ChatRequest):

    
    docs, confidence = retrieve_with_scores(request.question)

   
    answer = chat_chain.invoke(
        {
            "question": request.question,
        },
        config={"configurable": {"session_id": request.session_id}}
    )

    return {
        "session_id": request.session_id,
        "question": request.question,
        "answer": answer,
        "confidence": confidence
    }


@app.get("/")
def root():
    return {"message": "Advanced Chatbot API Running"}