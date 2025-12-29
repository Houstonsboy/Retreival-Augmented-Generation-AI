import requests
import os
from dotenv import load_dotenv

load_dotenv(override=True)
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# 1. CRITICAL: The URL should NOT contain the model name
url = "https://router.huggingface.co/hf-inference/v1/chat/completions"

# 2. CRITICAL: Exact Case-Sensitive Model ID
# Note: "Llama" is capitalized, "meta-llama" is lowercase
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

payload = {
    "model": MODEL_NAME,
    "messages": [
        {"role": "user", "content": "Hello! Testing the router."}
    ],
    "max_tokens": 100
}

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

resp = requests.post(url, json=payload, headers=headers)

if resp.status_code == 200:
    print("Success!")
    print(resp.json()['choices'][0]['message']['content'])
else:
    # This will help us see the exact error message from HF
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")