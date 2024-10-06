from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
import faiss  
import logging
import os
from split_document import split_document_into_chunks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

def generate_embeddings(chunks):
    """
    Generate embeddings for the given document chunks and store them using FAISS (uses the nearest neighbor search algorithm).
    
    Args:
        chunks (list): List of document chunks to generate embeddings for.
        
    Returns:
        FAISS index containing the document embeddings.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cpu"},  
        encode_kwargs={"normalize_embeddings": True}
    )
    logging.info(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded successfully.")

    knowledge_vector_database = FAISS.from_documents(
        chunks, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    logging.info("Embeddings generated successfully.")

    return knowledge_vector_database

def main():
    # Split the document into chunks
    file_path = "data/raw/TA-9-2024-0138_EN.pdf"
    chunks = split_document_into_chunks(file_path, chunk_size=256)

    # Generate embeddings for the document chunks
    knowledge_vector_database = generate_embeddings(chunks)

    # Ensure the embeddings directory exists
    embeddings_dir = "embeddings"
    os.makedirs(embeddings_dir, exist_ok=True)

    # Save the FAISS index to a file
    faiss_index_path = os.path.join(embeddings_dir, "faiss_index")
    faiss.write_index(knowledge_vector_database.index, faiss_index_path)
    logging.info(f"FAISS index saved successfully at '{faiss_index_path}'.")

if __name__ == "__main__":
    main()