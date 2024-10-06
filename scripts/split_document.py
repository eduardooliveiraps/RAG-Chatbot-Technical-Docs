from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

def split_document_into_chunks(file_path: str, chunk_size: int, tokenizer_name: str = EMBEDDING_MODEL_NAME):
    """
    Load a document and split it into smaller chunks for processing.

    Args:
        file_path (str): Path to the document file.
        chunk_size (int): The maximum size of each chunk (number of tokens).
        tokenizer_name (str): The name of the tokenizer to use for splitting the document.

    Returns:
        List of split document chunks.
    """
    # Check if the document file exists
    if not os.path.isfile(file_path):
        logging.error(f"The file '{file_path}' does not exist.")
        return None

    # Load the document using PyPDFLoader
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    logging.info(f"The document has been loaded successfully. Total number of pages: {len(pages)}.")

    # Initialize a text splitter 
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.1), # 10% overlap between chunks
        add_start_index=True, 
        strip_whitespace=True
    )

    chunks = text_splitter.split_documents(pages)
    logging.info(f"The document has been split into {len(chunks)} chunks.")

    return chunks




