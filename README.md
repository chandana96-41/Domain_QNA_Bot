# Local RAG Knowledge Assistant

A lightweight, fully local Retrieval-Augmented Generation (RAG) application that lets you upload your own documents, index them using local embeddings, and ask questions about them using a locally running LLM.

Built with **Python, Streamlit, Ollama, Qwen, ChromaDB, and LangChain**.

> **No cloud API keys. No external LLM API. Your documents stay on your machine.**

## Demo

Upload a PDF, Markdown file, text file, or DOCX document and ask questions about its contents.

The application:

1. Loads your documents
2. Splits them into smaller chunks
3. Converts chunks into embeddings
4. Stores the embeddings in ChromaDB
5. Retrieves the most relevant chunks for a question
6. Sends the retrieved context to a local Qwen model
7. Generates an answer grounded in the retrieved information
8. Displays the retrieved sources

### Architecture

```text
                ┌─────────────────┐
                │     Document    │
                │  PDF / TXT / MD │
                │      / DOCX     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Text Loader   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    Chunking     │
                │ 1000 characters │
                │  200 overlap    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Embeddings    │
                │ Ollama Embedding│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    ChromaDB     │
                │  Vector Store   │
                └────────┬────────┘
                         │
              User Question
                         │
                         ▼
                ┌─────────────────┐
                │    Retrieval    │
                │     Top-K       │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Retrieved       │
                │ Context + Query │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Qwen via       │
                │     Ollama      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │     Answer      │
                │   + Sources     │
                └─────────────────┘
```

## Features

- 📄 Upload multiple documents
- 🔍 Semantic search using embeddings
- 🧠 Local LLM inference with Ollama
- 🗃️ Persistent ChromaDB vector store
- 💬 Conversational Streamlit interface
- 📚 Source snippets for retrieved documents
- 🎛️ Adjustable Top-K retrieval
- 🔒 No external API required
- 🖥️ Designed to run on modest hardware

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| UI | Streamlit |
| LLM Runtime | Ollama |
| Generation Model | Qwen3 |
| Embeddings | Ollama `nomic-embed-text` |
| Vector Database | ChromaDB |
| RAG Framework | LangChain |
| PDF Processing | PyPDF |
| DOCX Processing | python-docx |

## How RAG Works

A normal LLM application looks like:

```text
Question → LLM → Answer
```

The model only has the information available in its prompt and learned parameters.

A RAG application introduces a retrieval step:

```text
Question
   ↓
Convert question to embedding
   ↓
Search vector database
   ↓
Retrieve relevant document chunks
   ↓
Add chunks to the prompt
   ↓
LLM
   ↓
Grounded answer
```

This allows the application to answer questions about information that was not part of the model's original training data.

## Requirements

Recommended:

- Python 3.10+
- Ollama
- 8 GB RAM or more
- Approximately 2 GB+ free disk space
- Windows, Linux, or macOS

The application is designed to work with small local models, so a dedicated GPU is not required.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama from:

https://ollama.com/

Make sure the Ollama service is running.

### 5. Download the models

For a lightweight setup:

```bash
ollama pull qwen3:0.6b
ollama pull nomic-embed-text
```

Verify:

```bash
ollama list
```

You should see both models.

### 6. Start the application

```bash
streamlit run app.py
```

The application should open in your browser.

## Using the Application

### Step 1 — Upload documents

Supported formats:

- PDF
- TXT
- Markdown
- DOCX

### Step 2 — Configure the models

The default configuration is:

```text
Embedding model: nomic-embed-text
Chat model: qwen3:0.6b
Top-K: 4
```

### Step 3 — Index the documents

Click:

```text
Index uploaded documents
```

The application extracts the text, creates chunks, generates embeddings, and stores them in ChromaDB.

### Step 4 — Ask questions

Use the chat box to ask questions about the uploaded documents.

The system retrieves relevant chunks and gives them to Qwen as context.

### Step 5 — Inspect the sources

Expand the **Sources** section below an answer to see which document chunks were retrieved.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
│
├── rag/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py
│   ├── chat.py
│   └── ollama_health.py
│
└── data/
    ├── uploads/
    └── chroma/
```

### `app.py`

Contains the Streamlit application and user interface.

### `rag/ingest.py`

Handles:

- Document loading
- Text extraction
- Chunking
- Embedding generation
- ChromaDB indexing

### `rag/chat.py`

Handles:

- Vector retrieval
- Context construction
- Prompt creation
- Qwen inference
- Source extraction

### `rag/config.py`

Contains model names, paths, and application configuration.

### `rag/ollama_health.py`

Checks whether Ollama is running and whether the required models are available.

## Chunking

Documents are split using a recursive character text splitter.

Current configuration:

```python
chunk_size = 1000
chunk_overlap = 200
```

The overlap helps preserve information that may span the boundary between two chunks.

Chunking is an important part of RAG quality. Poor chunks can lead to poor retrieval even when the LLM itself is capable.

## Retrieval

The application retrieves the top-K most relevant chunks from ChromaDB.

The number of retrieved chunks can be changed from the Streamlit sidebar.

For example:

```text
Top-K = 1
```

retrieves one chunk, while:

```text
Top-K = 4
```

retrieves four relevant chunks.

Increasing Top-K does not automatically make answers better. Too little context can omit important information, while too much irrelevant context can make generation less reliable.

## Grounding

The LLM is instructed to answer using the retrieved context rather than inventing information.

Conceptually:

```text
Context:
[Retrieved document chunks]

Question:
[User question]

        ↓

Qwen

        ↓

Answer based on context
```

If the answer cannot be found in the retrieved documents, the application instructs the model to say that there is not enough information.

### Important

RAG does **not** guarantee that hallucinations will disappear.

It provides the model with relevant external context, but retrieval errors, poor chunking, ambiguous questions, and model behavior can still produce incorrect answers.

## Local AI

This project uses Ollama to run the LLM locally.

That means the application does not need to send uploaded documents to a cloud LLM provider.

This makes the project useful for experimenting with:

- Private documents
- Offline applications
- Local AI
- Domain-specific assistants
- Privacy-sensitive workflows

## Limitations

This is intentionally a simple educational RAG implementation.

It currently does not implement:

- Hybrid search
- Reranking
- Advanced retrieval evaluation
- OCR for scanned PDFs
- Multimodal document understanding
- Query rewriting
- Production authentication
- Distributed vector databases
- Agentic workflows

These are possible directions for future development.

## Ideas for Extension

Try building your own version by adding:

- 🔖 Source citations
- 📑 Multiple PDF collections
- 🔎 Hybrid keyword + semantic search
- 🎯 Reranking
- 🧪 RAG evaluation
- 🖼️ Multimodal RAG
- 🌐 Web-page ingestion
- 💾 Conversation history
- 📊 Retrieval analytics
- 🔐 Authentication
- 🧑‍💻 Domain-specific assistants
- 🤖 Agentic workflows

## Learning Goals

This project is intended to help developers understand the fundamental building blocks behind modern RAG applications:

```text
LLM
 ↓
Local LLM
 ↓
Embeddings
 ↓
Vector Database
 ↓
Semantic Search
 ↓
Retrieval
 ↓
Context Augmentation
 ↓
RAG
 ↓
Application
```

The goal is not simply to build a chatbot, but to understand what happens between a user's question and the final answer.

## License

Add your preferred open-source license here.

## Acknowledgements

Built using open-source technologies including:

- Ollama
- Qwen
- LangChain
- ChromaDB
- Streamlit
- PyPDF

If you build something interesting with this project, consider opening an issue or submitting a pull request.