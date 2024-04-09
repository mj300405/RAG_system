from .preprocessing import Preprocessor
from .retrieval import Retriever
from .generation import PHI2Generator  
from sentence_transformers import SentenceTransformer

def main():
    """
    The main function orchestrates the Retrieval-Augmented Generation (RAG) process.
    
    It initiates the components necessary for the RAG system, including preprocessing,
    retrieval, and generation. Users can input queries to receive contextually relevant,
    generated answers. The loop continues until the user opts to exit.
    """
    # Initialize system components
    preprocessor = Preprocessor()
    retriever = Retriever(index_path="data/processed/index.faiss", dataset_path="data/processed/dataset")
    generator = PHI2Generator(num_threads=4)  # Adjust for CPU optimizations if CUDA isn't available
    model = SentenceTransformer('all-MiniLM-L6-v2')

    while True:  # Main interaction loop
        query = input("Enter your query: ")
        num_results_str = input("How many results would you like to retrieve? ")

        try:
            num_results = int(num_results_str)
        except ValueError:  # Ensure numerical input for the number of results
            print("Please enter a valid number for the results to retrieve.")
            continue

        # Preprocess the query to prepare for embedding and retrieval
        processed_query = preprocessor.preprocess_text(query)
        
        # Generate an embedding for the processed query
        query_embedding = model.encode(processed_query)

        # Use the retriever to find relevant documents based on the query embedding
        distances, indices = retriever.search(query_embedding, k=num_results)
        retrieved_data = retriever.get_retrieved_chunks(indices)

        # Combine the texts of retrieved documents to form a single context string
        context = " ".join([data['text'] for data in retrieved_data])

        # Generate an answer based on the combined context
        answer = generator.generate_answer(context)

        # Display the retrieved information and the generated answer
        print(f"Distance scores:\n{distances}")
        print(f"Generated Answer:\n{answer}")

        # Prompt the user to continue or exit the loop
        continue_response = input("Do you want to continue? (yes/no): ")
        if continue_response.lower() not in ['yes', 'y']:
            break  # Exit condition

if __name__ == "__main__":
    main()
