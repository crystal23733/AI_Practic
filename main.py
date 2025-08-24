from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")

client = OpenAI(api_key=api_key)

response = client.responses.create(
  model="gpt-5",
  input="스티븐 호킹이 누구야?"
)

print(response.output_text)