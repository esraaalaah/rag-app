import os
from dotenv import load_dotenv
from openai import OpenAI

# تحميل ملف .env
load_dotenv()

# قراءة المفتاح
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("⚠️ مفيش مفتاح OpenRouter متسجل! ضيفي OPENROUTER_API_KEY في .env")

# تعريف العميل باستخدام OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

import json

def chat_json(messages, max_tokens=2200, temperature=0.4, model="openai/gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)   # يحول النص لـ dict
    except json.JSONDecodeError:
        return {"questions": [content]}  # fallback لو مش JSON
