from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def get_vision_llm():
    return ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    api_key=os.getenv("GROQ_API_KEY")
)

llm = get_vision_llm()

system_message = SystemMessage(content="""You are given a screenshot and the DOM of a webpage. Your task is to analyze the screenshot and DOM and guide the user on how to interact with the webpage to achieve their goal. The user will ask you questions about how to use the webpage, and you will provide step-by-step instructions based on the screenshot and DOM.""")

history = []


def ask_question(question, screenshot, dom):
    user_message = HumanMessage(
        content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"}},
            {"type": "text", "text": f"DOM: {dom}"},
            {"type": "text", "text": question},
        ]
    )
    messages = [system_message, user_message]
    response = llm.invoke(messages + history)
    
    return response