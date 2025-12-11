from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(
    api_key=os.getenv("CLAUDE_AGENT"),
    base_url="https://agentrouter.org/v1"
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str
    max_tokens: int = 2000
    temperature: float = 0.1

@app.post("/chat")
def chat(req: ChatRequest):
    response = client.chat.completions.create(
        model=req.model,
        messages=[m.dict() for m in req.messages],
        max_tokens=req.max_tokens,
        temperature=req.temperature
    )
    return {
        "response": response.choices[0].message["content"]
    }
