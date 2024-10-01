from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_and_split_pdf(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Load a PDF document and split it into smaller chunks for processing.
    
    Args:
        pdf_path (str): Path to the PDF file.
        chunk_size (int): The maximum size of each chunk (number of characters).
        chunk_overlap (int): Overlap between chunks to maintain context.
        
    Returns:
        List of split documents.
    """
    # Check if the PDF file exists
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"The file '{pdf_path}' does not exist.")
    
    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # Load the PDF as an array of page documents
    
    # Initialize a text splitter with recursive character splitting strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,       # Maximum size of each chunk (number of characters)
        chunk_overlap=chunk_overlap   # Overlap between chunks to maintain context
    )
    
    # Split the pages into smaller chunks
    docs = text_splitter.split_documents(pages)
    
    return docs

def main():
    pdf_path = "data/raw/TA-9-2024-0138_EN.pdf"  # Path to the PDF file
    split_docs = load_and_split_pdf(pdf_path)

    # Print the number of chunks and sample text from the first chunk
    print(f"Total number of chunks: {len(split_docs)}")
    print(f"First chunk content: {split_docs[0].page_content[:300]}")  # Preview the first 300 characters of the first chunk

if __name__ == "__main__":
    main()
