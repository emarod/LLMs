# 🤖 LlamaIndex RAG Demo

This is a (toy) lightweight Retrieval-Augmented Generation (RAG) project using **LlamaIndex** to enable AI-powered question answering over custom documents.

---

## 🔧 Requirements

Before running the project, make sure you have the following installed:

```bash
pip install streamlit
pip install llama-index
```

> 🧩 *Other dependencies may be installed automatically when you run the scripts.*

---

## 📦 Project Structure

### 💻 UI Version *(Still in Development)*

Scripts:
- `App_No_Web_Scrapp_Local.py`  
- `App_No_Web_Scrapp_Pinecone.py`  
- `App_Web_Scrapping.py`  

Run UI apps with:

```bash
streamlit run <script_name>.py
```

⚠️ **Important**  
UI versions reload documents (or perform web scraping) on every user prompt:
- This means a new Pinecone call on every message in `App_No_Web_Scrapp_Pinecone.py`
- Or a new web request per prompt in `App_Web_Scrapping.py`

🔻 As a result, performance may be poor for now. Optimization is in progress!

---

### 🖥️ Console Version *(Fully Functional)*

Scripts:
- `Chat_(Chat Engine).py`  
- `Chat_(Query Engine).py`  

📌 Differences:
- `Chat Engine`: Maintains conversational memory during execution (chatbot-like)
- `Query Engine`: Executes single-shot queries (no memory between prompts)

Run console apps with:

```bash
python <script_name>.py
```

---

## 📂 Add Your Own Data

To use RAG with your own files, simply place `.pdf` or `.txt` or `.md` files inside the `/data` folder.

---

## 🔐 API Keys & Environment Setup

You’ll need the following API keys:

- `LLAMA_CLOUD_API_KEY`
- `GOOGLE_API_KEY`

Set them in your Python script like this:

```python
import os

os.environ["LLAMA_CLOUD_API_KEY"] = "your_llama_key"
os.environ["GOOGLE_API_KEY"] = "your_google_key"
```

Or store them in a `.env` file and load with:

```python
from dotenv import load_dotenv
load_dotenv()
```

For this example we are using Gemini AI, but you can use any other LLM supported by llama-index framework such as OPEN AI or Mistral AI
Just check out the documentation !

---

## 🔗 Optional Services

### 🧠 Pinecone (Vector Database)

```python
from pinecone import Pinecone
pinecone_client = Pinecone(api_key="your_pinecone_key")
```

### 🌐 HyperBrowser (Web Scraping)

```python
from hyperbrowser import HyperBrowser
client = HyperBrowser(api_key="your_hb_key")
```

---

## 📚 Learn More

Explore more services and integrations available with LlamaIndex:  
👉 [LlamaIndex Documentation](https://docs.llamaindex.ai/en/stable/api_reference/)
