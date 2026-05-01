from google import genai

# Replace with your Gemini API key
client = genai.Client(api_key="API_key")

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="Explain college admission process in simple words."
)

print(response.text)
