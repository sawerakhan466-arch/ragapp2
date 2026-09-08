# 📘 HR Policy Assistant

An AI-powered HR Policy Assistant built with Retrieval-Augmented Generation (RAG).

The application allows users to upload an HR policy PDF and ask natural-language questions about the policy.

## 🚀 Features

- Upload HR Policy PDF
- Extract text using PyMuPDF
- Split documents into overlapping chunks
- Generate embeddings using Sentence Transformers
- Store embeddings in FAISS
- Retrieve relevant policy sections
- Generate answers using Groq
- Uses OpenAI GPT-OSS 20B through Groq
- Shows retrieved source pages
- Displays similarity scores
- Maintains chat history
- Streamlit web interface
- No database required

## 🧠 RAG Architecture

```text
HR Policy PDF
      ↓
PyMuPDF
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
Sentence Transformers
      ↓
Vector Embeddings
      ↓
FAISS
      ↓
User Question
      ↓
Question Embedding
      ↓
Top-K Relevant Chunks
      ↓
Groq GPT-OSS 20B
      ↓
HR Policy Answer
