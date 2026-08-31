import requests
import json

print("\n--- SENDING REQUEST ---")
query = "A political text arguing that society requires a strong, absolute sovereign to avoid the 'war of all against all' and ensure order."
print(f"User: {query}\n")

payload = {"message": query}
response = requests.post("http://127.0.0.1:7860/chat", json=payload)
data = response.json()

print("--- AI RESPONSE ---")
print(data.get('answer', 'Error'))
print("\n")