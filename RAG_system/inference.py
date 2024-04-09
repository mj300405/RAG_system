from .preprocessing import Preprocessor
from .retrieval import Retriever
from .generation import PHI2Generator  
from sentence_transformers import SentenceTransformer

def main():
    preprocessor = Preprocessor()
    retriever = Retriever(index_path="data/processed/index.faiss", dataset_path="data/processed/dataset")
    generator = PHI2Generator(num_threads=4)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    while True:  # Keeps the loop running until the user decides to exit
        query = input("Enter your query: ")
        num_results_str = input("How many results would you like to retrieve? ")

        try:
            num_results = int(num_results_str)
        except ValueError:  # In case of non-integer input for num_results
            print("Please enter a valid number for the results to retrieve.")
            continue

        # Preprocess the query
        processed_query = preprocessor.preprocess_text(query)
        
        # Generate query embedding
        query_embedding = model.encode(processed_query)

        # Perform search with the retriever
        distances, indices = retriever.search(query_embedding, k=num_results)
        retrieved_data = retriever.get_retrieved_chunks(indices)

        # Combine the texts of the retrieved documents to form the context
        context = " ".join([data['text'] for data in retrieved_data])

        # Generate an answer based on the combined context
        answer = generator.generate_answer(context)

        # Display the generated answer
        print(f"Distance scores:\n{distances}")
        print(f"Generated Answer:\n{answer}")

        # Ask if the user wants to continue or exit
        continue_response = input("Do you want to continue? (yes/no): ")
        if continue_response.lower() not in ['yes', 'y']:
            break  # Exit the loop if the user does not want to continue

if __name__ == "__main__":
    main()
