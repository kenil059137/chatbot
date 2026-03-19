import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(
    os.getenv("MONGO_URL", "mongodb://localhost:27017/"),
    serverSelectionTimeoutMS=3000
)
db = client["charusat_chatbot"]
collection = db["chat_history"]

# index for faster queries
collection.create_index("session_id")


def save_message(session_id: str, role: str, message: str):
    collection.insert_one({
        "session_id": session_id,
        "role": role,
        "message": message,
        "timestamp": datetime.now()
    })


def get_history(session_id: str, limit: int = 10):
    messages = collection.find(
        {"session_id": session_id},
        sort=[("timestamp", 1)]
    ).limit(limit)

    history = ""
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['message']}\n"

    return history.strip()


def get_all_sessions():
    return collection.distinct("session_id")


def delete_session(session_id: str):
    collection.delete_many({"session_id": session_id})