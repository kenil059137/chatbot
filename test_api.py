import os
from dotenv import load_dotenv
import google.genai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("API key not found in .env")
    exit(1)

try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Hello, test message"
    )
    print("API key works! Response:", response.text)
except Exception as e:
    print("API key error:", str(e))