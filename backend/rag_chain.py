import os
import ssl
from dotenv import load_dotenv

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

from google import genai
from groq import Groq

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def _call_gemini(prompt):
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text


def _call_groq(prompt):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # current active model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response.choices[0].message.content


PROVIDERS = [
    ("Gemini", _call_gemini),
    ("Groq",   _call_groq),
]


def gemini_llm(prompt):
    if hasattr(prompt, "to_string"):
        prompt = prompt.to_string()

    for name, caller in PROVIDERS:
        try:
            print(f"Trying {name}...")
            result = caller(prompt)
            print(f"{name} succeeded!")
            return result
        except Exception as e:
            print(f"{name} failed: {e} — trying next...")

    return "I'm sorry, all AI services are currently unavailable. Please try again later."