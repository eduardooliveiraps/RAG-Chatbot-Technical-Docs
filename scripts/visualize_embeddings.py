import faiss  
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_faiss_index(index_path):
    """
    Load the FAISS index from the specified file path.
    
    Args:
        index_path (str): Path to the FAISS index file.
        
    Returns:
        FAISS index object.
    """
    if not os.path.exists(index_path):
        logging.error(f"FAISS index file not found at '{index_path}'")
        return None
    
    index = faiss.read_index(index_path)
    logging.info(f"FAISS index loaded successfully from '{index_path}'")
    return index

def visualize_embeddings(index):
    """
    Visualize the embeddings using PCA and Matplotlib.
    
    Args:
        index (faiss.Index): The FAISS index containing the embeddings.
    """
    # Extract embeddings from the FAISS index
    embeddings = index.reconstruct_n(0, index.ntotal)
    
    # Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Plot the embeddings
    plt.figure(figsize=(10, 10))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=5)
    plt.title("PCA visualization of embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

def main():
    # Path to the saved FAISS index file
    faiss_index_path = "embeddings/faiss_index"

    # Load the FAISS index
    index = load_faiss_index(faiss_index_path)
    if index is None:
        return

    # Visualize the embeddings
    visualize_embeddings(index)

if __name__ == "__main__":
    main()