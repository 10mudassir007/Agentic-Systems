from smolagents import LiteLLMModel
from smolagents import CodeAgent

messages = [
  {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}
]

model_id="groq/llama-3.3-70b-versatile"

agent = CodeAgent(tools=[], model = LiteLLMModel(model_id="groq/llama-3.3-70b-versatile", temperature=0.2), add_base_tools=True)

print(agent.run("Why does Mike not know many people in New York?",
    additional_args={"mp3_sound_file_url":'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/recording.mp3'}
))