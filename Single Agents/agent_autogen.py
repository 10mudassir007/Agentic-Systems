import gradio as gr
from dotenv import load_dotenv
import os
from autogen import ConversableAgent
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# LLM Configuration for Groq
config_list = [
    {
        "model": "qwen-2.5-32b",
        "api_type": "groq"
    }
]

# Define Conversable Agent
assistant = ConversableAgent(
    name="Assistant",
    system_message="You are a helpful AI assistant.",
    llm_config={"config_list": config_list},
)

# Tavily Client for Web Search
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Function to Handle Chat
def chat_with_agent(user_query):
    chat_result = assistant.initiate_chat(assistant, message=user_query, max_turns=3)
    reply = next(
        (msg["content"] for msg in chat_result.chat_history if msg.get("name") == "Assistant"),
        "I couldn't generate a response."
    )
    return reply

# Function to Handle Web Search
def search_web(query):
    results = tavily_client.search(query, max_results=3)
    if results and "results" in results and results["results"]:
        return "\n\n".join([r["content"] for r in results["results"]])
    return "No relevant results found online."

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 AI Chatbot with Web Search")
    
    with gr.Tab("Chat"):
        chatbot = gr.ChatInterface(chat_with_agent)

    with gr.Tab("Search"):
        search_input = gr.Textbox(label="Enter your search query:")
        search_output = gr.Textbox(label="Search Results", interactive=False)
        search_button = gr.Button("Search")
        search_button.click(fn=search_web, inputs=search_input, outputs=search_output)

# Launch Gradio App
demo.launch()
