from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from backend.services.multi_agent_rag import multi_agent_rag
from backend.services.chat_history import save_message, get_history, get_all_sessions

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
        history = get_history(request.session_id)

        answer, confidence = multi_agent_rag(
            question=request.question,
            session_id=request.session_id,
            history=history
        )

        save_message(request.session_id, "user", request.question)
        save_message(request.session_id, "assistant", answer)

        return {
            "session_id": request.session_id,
            "question": request.question,
            "answer": answer,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{session_id}")
def get_chat_history(session_id: str):
    history = get_history(session_id, limit=50)
    return {"session_id": session_id, "history": history}


@app.get("/sessions")
def get_sessions():
    sessions = get_all_sessions()
    return {"sessions": sessions}


@app.get("/")
def root():
    return {"message": "Advanced Chatbot API Running"}