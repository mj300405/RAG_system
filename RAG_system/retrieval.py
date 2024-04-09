import numpy as np
import faiss
from datasets import load_from_disk

class Retriever:
    """
    A class for retrieving relevant document chunks based on query embeddings using a FAISS index.
    
    This class encapsulates functionality for searching a pre-built FAISS index to find
    the most relevant document chunks from a dataset based on the semantic similarity
    of embeddings.
    
    Attributes:
        faiss_index (faiss.Index): The loaded FAISS index for similarity search.
        hf_dataset (datasets.Dataset): The dataset loaded from disk, containing the document chunks.
        
    Args:
        index_path (str): The file path to the saved FAISS index.
        dataset_path (str): The file path to the saved Hugging Face dataset.
    """
    
    def __init__(self, index_path, dataset_path):
        """
        Initializes the Retriever class by loading the FAISS index and Hugging Face dataset from disk.
        """
        self.faiss_index = faiss.read_index(index_path)
        self.hf_dataset = load_from_disk(dataset_path)

    def search(self, query_embedding, k=5):
        """
        Searches the FAISS index for the k most similar document chunks based on the query embedding.
        
        Args:
            query_embedding (np.ndarray): The embedding vector of the query.
            k (int, optional): The number of similar document chunks to retrieve. Defaults to 5.
        
        Returns:
            tuple: A tuple containing two elements:
                - distances (np.ndarray): The distances of the retrieved document chunks from the query.
                - indices (np.ndarray): The indices of the retrieved document chunks in the dataset.
        """
        # Convert query_embedding to the expected shape and dtype for FAISS search
        query_embedding = np.array([query_embedding]).astype("float32")
        
        # Perform the search in the FAISS index
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # Check if any results were found
        if indices.size == 0:
            print("No similar documents found.")
            return None, None
        
        return distances, indices

    def get_retrieved_chunks(self, indices):
        """
        Retrieves the document chunks corresponding to the given indices from the dataset.
        
        Args:
            indices (np.ndarray): The indices of the document chunks to retrieve.
        
        Returns:
            list: A list of retrieved document chunks.
        """
        if indices is None:
            return []
        
        retrieved_chunks = []
        for idx in indices[0]:  # Loop through the first dimension of indices
            try:
                retrieved_chunks.append(self.hf_dataset[int(idx)])
            except Exception as e:
                print(f"Error retrieving document at index {idx}: {e}")
        return retrieved_chunks
