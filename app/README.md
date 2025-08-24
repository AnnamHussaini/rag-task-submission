# Local RAG System with Citations

## Overview
This is a Retrieval-Augmented Generation (RAG) pipeline built for the AI Developer Intern task.  
It ingests PDFs (native + scanned), builds a local vector index, and answers questions with **citations** and **highlighted evidence**.

## Features
- PDF text extraction with OCR fallback (PyMuPDF + Tesseract).
- Embeddings using `BAAI/bge-small-en-v1.5` (Sentence Transformers).
- Local vector database with ChromaDB.
- Local LLM generation using Ollama (`llama3.2`).
- Evidence highlighting and citations (source + page number).
- CLI interface for ingestion and Q&A.

## How to run
1. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
