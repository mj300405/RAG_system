import numpy as np
import faiss
from datasets import load_from_disk

class Retriever:
    def __init__(self, index_path, dataset_path):
        self.faiss_index = faiss.read_index(index_path)
        self.hf_dataset = load_from_disk(dataset_path)

    def search(self, query_embedding, k=5):
        # Ensure query_embedding is in the right shape
        query_embedding = np.array([query_embedding]).astype("float32")
        
        # Perform the search
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # Handle the case where no results are found
        if indices.size == 0:
            print("No similar documents found.")
            return None, None
        
        return distances, indices

    def get_retrieved_chunks(self, indices):
        if indices is None:
            return []
        
        retrieved_chunks = []
        for idx in indices[0]:  # Assuming indices is a 2D array with shape (1, k)
            try:
                retrieved_chunks.append(self.hf_dataset[int(idx)])
            except Exception as e:
                print(f"Error retrieving document at index {idx}: {e}")
        return retrieved_chunks
