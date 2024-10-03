from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_pdf_into_chunks(file_path, chunk_size=512, chunk_overlap=100):
    """
    Load a PDF document and split it into smaller chunks for processing.
    
    Args:
        file_path (str): Path to the PDF file.
        chunk_size (int): The maximum size of each chunk (number of characters).
        chunk_overlap (int): Overlap between chunks to maintain context.
        
    Returns:
        List of split document chunks.
    """
    # Check if the PDF file exists
    if not os.path.isfile(file_path):
        logging.error(f"The file '{file_path}' does not exist.")
        return None
    
    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader(file_path)
    lazy_pages = loader.lazy_load()
    
    pages = [page for page in lazy_pages]  # Load all pages into memory
    logging.info(f"The PDF document has been loaded successfully. Total number of pages: {len(pages)}.")
    
    # Initialize a text splitter with recursive character splitting strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,       # Maximum size of each chunk (number of characters)
        chunk_overlap=chunk_overlap  # Overlap between chunks to maintain context
    )
    
    # Split the pages into smaller chunks
    chunks = text_splitter.split_documents(pages)
    logging.info(f"The PDF document has been split into {len(chunks)} chunks.")
    
    return chunks

def main():
    file_path = "data/raw/TA-9-2024-0138_EN.pdf"  # Path to the PDF file
    split_chunks = split_pdf_into_chunks(file_path)

if __name__ == "__main__":
    main()
