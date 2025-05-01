import os
import json
import streamlit as st
from dotenv import load_dotenv
from hyperbrowser import AsyncHyperbrowser as Hyperbrowser
from hyperbrowser.models.scrape import StartScrapeJobParams

from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Document, StorageContext, VectorStoreIndex, SummaryIndex
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Settings

import nest_asyncio
nest_asyncio.apply()

os.environ["LLAMA_CLOUD_API_KEY"] = "llx-kAgbmQAx4V3JBDHr5RRriKvHFdXXYiQvQ7uK8D59FFoxERKS"
os.environ["GOOGLE_API_KEY"] = "AIzaSyA2lQuATkXbD4XNG-3kGN9qm41DlZnS9ZU"
client = Hyperbrowser(api_key="hb_a52887542ea603b1ed68480f4f77")
pinecone_client = Pinecone(api_key="pcsk_2AA7Gh_Ev537s93XkMcyyKFqKZn4rxuwVVmpAsu1eqRDRPnk6xhcKSRhNZUve6c5aMNagT")

urls_for_scrapping = {
        "Alumnos Redes y Ciencia de Datos: " : "https://www.cic.ipn.mx/index.php/alumnos-rycd"
        }
        
load_dotenv()

# Initialize Hyperbrowser client

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
        result_type="markdown"  # "markdown" and "text" are available
    )

    # use SimpleDirectoryReader to parse our file
    file_extractor = {".pdf": parser}
    text_pdf = SimpleDirectoryReader(input_files=['./data/carga_academica_2025.pdf'], file_extractor=file_extractor).load_data()

    documents = [Document(text=t.text) for t in text_pdf]

    for key, url in urls_for_scrapping.items():
        #Start scraping and wait for completion
        doc = await client.scrape.start_and_wait(
            StartScrapeJobParams(url=url)
        )

        data_scrapped = key + " " + doc.data.markdown

        documents.append(Document(text = data_scrapped))

    # Index Setup

    pinecone_index = pinecone_client.Index("demo-cic")

    # Create a PineconeVectorStore using the specified pinecone_index
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    # Create a StorageContext using the created PineconeVectorStore
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # Use the chunks of documents and the storage_context to create the index
    index = SummaryIndex.from_documents(
        documents, 
        storage_context=storage_context
    )

    #index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # Index Setup

    query_engine = index.as_query_engine()

    # Query the index, send the context to Gemini, and wait for the response
    # gemini_response = query_engine.query("Give the name of a student that starts with letter C:")

    st.title("🤖 Chatbot del CIC IPN")
    user_input = st.text_input("Escribe tu pregunta:")

    if user_input:
        respuesta = query_engine.query(user_input)
        st.write("🤖 Bot:", respuesta)

    # print(gemini_response)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())