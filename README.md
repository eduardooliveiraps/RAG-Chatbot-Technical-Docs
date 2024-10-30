# Building a RAG Chatbot for Technical Documentation

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) chatbot to answer questions about technical documentation. The chosen document for this demonstration is the [Artificial Intelligence Act](https://www.europarl.europa.eu/doceo/document/TA-9-2024-0138_EN.pdf), recently passed by the European Parliament.

## Project Overview

Our approach follows the steps below:
1. Split the document into chunks of text
2. Generate and store the embeddings for each chunk
3. Create a retriever model to find the most relevant chunks for a given question
4. Initialize the LLM and prompt template
5. Define RAG chain
5. Invoke RAG chain

## Pre-requisites

To reproduce this project, ensure you have Python installed on your system. Then, install the required libraries:

```bash
pip install transformers
pip install torch
pip install faiss-cpu
pip install langchain
pip install langchain_huggingface
pip install streamlit
pip install streamlit_chat
```

## Usage Instructions

