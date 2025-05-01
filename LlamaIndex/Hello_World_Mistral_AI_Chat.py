import os
from llama_index.core.llms import ChatMessage
from llama_index.llms.mistralai import MistralAI

os.environ["MISTRAL_API_KEY"] = "w3EKHft6JgsYPakUgX8WBsO98NTj3ZiP"

messages = [
    ChatMessage(role="system", content="Eres una persona de México"),
    ChatMessage(role="user", content="Cuéntame sobre tu día a día"),
]
resp = MistralAI().chat(messages)

print(resp)