from llama_index.llms.mistralai import MistralAI

# To customize your API key, do this
# otherwise it will lookup MISTRAL_API_KEY from your env variable
# llm = MistralAI(api_key="")

llm = MistralAI(api_key="w3EKHft6JgsYPakUgX8WBsO98NTj3ZiP")

resp = llm.complete("Paul Graham is ")
print(resp)