from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from backend.services.multi_agent_rag import multi_agent_rag

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
    try:
        answer, confidence = multi_agent_rag(
            question=request.question,
            session_id=request.session_id
        )

        return {
            "session_id": request.session_id,
            "question": request.question,
            "answer": answer,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Advanced Chatbot API Running"}