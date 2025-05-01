import os
import json
import streamlit as st

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Document, StorageContext, VectorStoreIndex, SummaryIndex
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings

import asyncio
import nest_asyncio

os.environ["LLAMA_CLOUD_API_KEY"] = "llx-kAgbmQAx4V3JBDHr5RRriKvHFdXXYiQvQ7uK8D59FFoxERKS"
os.environ["GOOGLE_API_KEY"] = "AIzaSyA2lQuATkXbD4XNG-3kGN9qm41DlZnS9ZU"

async def main():

    llm = GoogleGenAI(
        model="gemini-2.0-flash",
    )
    embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004")

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512

    documents = []
    # set up parser
    parser = LlamaParse(
        result_type="text"  # "markdown" and "text" are available
    )

    # use SimpleDirectoryReader to parse our file
    file_extractor = {".pdf": parser, ".txt" : parser}
    text_pdf = SimpleDirectoryReader(input_dir='./data', file_extractor=file_extractor).load_data()

    documents = [Document(text=t.text) for t in text_pdf]

    # Index Setup

    # # Use the chunks of documents and the storage_context to create the index
    index = VectorStoreIndex.from_documents(documents)

    # Index Setup

    query_engine = index.as_query_engine()

    # Query the index, send the context to Gemini, and wait for the response
    # gemini_response = query_engine.query("Give the name of a student that starts with letter C:")

    st.title("🤖 Chatbot del CIC IPN")
    user_input = st.text_input("Escribe tu pregunta:")

    if user_input:
        respuesta = query_engine.query(user_input)
        st.write("🤖 Bot:\n", respuesta.response)

    # print(gemini_response)


if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())