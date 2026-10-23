import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env from the same folder as this script
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GOOGLE_API_KEY")

# Check whether the key was loaded
if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY is missing. "
        "Check your .env file."
    )

print("API key loaded successfully!")

try:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=api_key,
        temperature=0.1,
        timeout=60,
        max_retries=0
    )

    response = llm.invoke(
        "What is cybersecurity? Explain in two sentences."
    )

    print("Gemini Response:")
    print(response.content)

except Exception as e:
    print("Error type:", type(e).__name__)
    print("Error:", str(e))