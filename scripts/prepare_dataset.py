import pandas as pd
from RAG_system.preprocessing import Preprocessor
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import tqdm

def prepare_dataset(csv_path='data/raw/medium.csv', dataset_path='./data/processed/dataset', index_path='./data/processed/index.faiss'):
    """
    Prepares and indexes a dataset for retrieval-augmented generation tasks.
    
    This function reads article data from a CSV file, preprocesses the text, 
    generates embeddings for each chunk of the articles, and creates a FAISS index 
    for efficient similarity search. The processed data and embeddings are saved 
    to disk for future use.
    
    Args:
        csv_path (str): Path to the raw CSV file containing the articles.
        dataset_path (str): Path where the processed dataset should be saved.
        index_path (str): Path where the FAISS index should be saved.
    """
    # Load and preprocess the dataset
    df = pd.read_csv(csv_path)
    preprocessor = Preprocessor()

    # Initialize the sentence transformer model for embedding generation
    model = SentenceTransformer('all-MiniLM-L6-v2')

    processed_data = []  # Stores preprocessed text data
    embeddings = []  # Stores embeddings for indexing

    # Process each article in the dataset
    for _, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        title = row['Title']
        # Break articles into manageable chunks
        text_chunks = preprocessor.chunk_article(row['Text'])

        for chunk in text_chunks:
            # Generate and store embeddings for each chunk
            embedding = model.encode(chunk)
            embeddings.append(embedding)
            
            # Store processed data for dataset creation
            processed_data.append({
                'title': title,
                'text': chunk,
                'embeddings': embedding.tolist()  # Convert embedding to list for serialization
            })

    # Create a DataFrame from the processed data
    processed_df = pd.DataFrame(processed_data)

    # Convert the DataFrame to a Hugging Face Dataset and save to disk
    hf_dataset = Dataset.from_pandas(processed_df)
    hf_dataset.save_to_disk(dataset_path)

    # Create and populate the FAISS index with article chunk embeddings
    d = len(embeddings[0])  # Determine embedding dimensionality
    index = faiss.IndexFlatL2(d)  # Initialize FAISS index with L2 distance metric
    faiss_embeddings = np.array(embeddings).astype('float32')  # Convert embeddings to the correct dtype
    index.add(faiss_embeddings)  # Add embeddings to the FAISS index
    faiss.write_index(index, index_path)  # Save the index to disk for later retrieval

if __name__ == '__main__':
    prepare_dataset()
