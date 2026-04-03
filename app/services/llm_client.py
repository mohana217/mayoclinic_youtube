from langchain_openai import ChatOpenAI
from app.config.settings import OPENAI_API_KEY, MODEL_NAME

import os
from langchain_openai import ChatOpenAI

def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing")

    return ChatOpenAI(
        model="gpt-4o-mini",   
        temperature=0,
        api_key=api_key
    )
    