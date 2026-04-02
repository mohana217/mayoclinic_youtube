from langchain_openai import ChatOpenAI
from app.config.settings import OPENAI_API_KEY, MODEL_NAME

def get_llm():
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    )