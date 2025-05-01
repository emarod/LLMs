import os
import asyncio
import nest_asyncio

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Document, SummaryIndex
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings

# Configurar API Keys
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-kAgbmQAx4V3JBDHr5RRriKvHFdXXYiQvQ7uK8D59FFoxERKS"
os.environ["GOOGLE_API_KEY"] = "AIzaSyA2lQuATkXbD4XNG-3kGN9qm41DlZnS9ZU"

def initialize_chatbot():
    """Carga documentos y crea el índice solo una vez."""
    
    print("🔄 Cargando documentos e indexando...")

    # Configurar modelos
    llm = GoogleGenAI(model="gemini-2.0-flash")
    embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004")
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512

    # Configurar parser y cargar documentos
    parser = LlamaParse(result_type="text")
    file_extractor = {".pdf": parser, ".txt": parser}
    text_pdf = SimpleDirectoryReader(input_dir="./data", file_extractor=file_extractor).load_data()
    
    # Convertir a formato adecuado
    documents = [Document(text=t.text) for t in text_pdf]

    # Crear índice en memoria
    index = SummaryIndex.from_documents(documents)

    # Crear motor de consulta
    chat_engine = index.as_chat_engine(chat_mode="context")

    print("✅ Chatbot listo. Escribe 'salir' para terminar.\n")
    return chat_engine

async def chat_loop(chat_engine):
    """Bucle de conversación en la terminal."""
    while True:
        user_input = input("🗣️ Tú: ")
        if user_input.lower() == "salir":
            print("🤖 Chatbot: ¡Hasta luego! 👋")
            break
        
        response = await chat_engine.achat(user_input)
        print(f"🤖 Chatbot:\n {response.response}\n")

# Permitir ejecución de eventos async
if __name__ == "__main__":
    nest_asyncio.apply()
    chat_engine = initialize_chatbot()  # Carga el índice una sola vez
    asyncio.run(chat_loop(chat_engine))  # Inicia el chatbot