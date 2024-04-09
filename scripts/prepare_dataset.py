import pandas as pd
from RAG_system.preprocessing import Preprocessor
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import tqdm

def prepare_dataset(csv_path='data/raw/medium.csv', dataset_path='./data/processed/dataset', index_path='./data/processed/index.faiss'):
    df = pd.read_csv(csv_path)
    preprocessor = Preprocessor()

    # Initialize the sentence transformer model for embedding generation
    model = SentenceTransformer('all-MiniLM-L6-v2')

    processed_data = []
    embeddings = []  # To store embeddings for the FAISS index

    for _, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        title = row['Title']
        text_chunks = preprocessor.chunk_article(row['Text'])

        for chunk in text_chunks:
            # Generate an embedding for the chunk
            embedding = model.encode(chunk)
            embeddings.append(embedding)
            
            # Store the title, chunk (as text), and embedding (as list for JSON serializability)
            processed_data.append({
                'title': title,
                'text': chunk,
                'embeddings': embedding.tolist()
            })

    # Convert the processed data into a DataFrame
    processed_df = pd.DataFrame(processed_data)

    # Convert the processed DataFrame into a Hugging Face Dataset and save to disk
    hf_dataset = Dataset.from_pandas(processed_df)
    hf_dataset.save_to_disk(dataset_path)

    # Now, create and populate the FAISS index with embeddings
    d = len(embeddings[0])  # Dimensionality of the embeddings
    index = faiss.IndexFlatL2(d)  # Using L2 distance for the similarity measure
    faiss_embeddings = np.array(embeddings).astype('float32')  # Make sure to convert embeddings list to a NumPy array of type float32
    index.add(faiss_embeddings)  # Add embeddings to the index
    faiss.write_index(index, index_path)  # Save the index to disk

if __name__ == '__main__':
    prepare_dataset()
