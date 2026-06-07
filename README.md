# Telegram RAG Assistant

## Overview

A Telegram-based AI assistant that allows users to upload PDF documents, automatically indexes them using vector embeddings, and answers questions using Retrieval-Augmented Generation (RAG).

## Features

* Upload PDFs through Telegram
* Automatic PDF ingestion
* ChromaDB vector database
* Multi-document collections
* Switch between collections
* Semantic search
* Gemini-powered responses

## Tech Stack

* Python
* Telegram Bot API
* Gemini API
* ChromaDB
* Sentence Transformers
* PyPDF

## Architecture

PDF Upload
→ Text Extraction
→ Chunking
→ Embeddings
→ ChromaDB
→ Retrieval
→ Gemini
→ Telegram Response

## Commands

/start

/collections

/use <collection_name>

## Example Workflow

1. Upload a PDF.
2. Bot automatically indexes the document.
3. Run `/collections` to view available documents.
4. Run `/use document_name`.
5. Ask questions about the document.

## Future Improvements

* Better PDF parsing
* User-specific collections
* Web interface
* OCR support
* Cloud deployment

