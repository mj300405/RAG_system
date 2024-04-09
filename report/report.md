# RAG system report

## Introduction

This report presents the design and functionality of a Retrieval-Augmented Generation (RAG) system that synthesizes answers from a text corpus. The system merges cutting-edge natural language processing techniques to deliver contextually relevant responses to user queries.

## System Design
The designed system incorporates a synergistic interaction between a retrieval engine and a generative model to efficiently process and synthesize information from a large corpus of text. Here’s an in-depth look at its architecture:

- **Retrieval Engine**: At the core of the retrieval component is the FAISS library, renowned for its efficient similarity search and clustering of dense vectors. This engine processes queries by converting them into embeddings and then swiftly retrieves the most semantically similar text fragments from the indexed dataset. The precision of this component is critical for the relevance of information fed into the generative model.

- **Generative Model**: The PHI2Generator, leveraging the "phi-1_5" model, is tasked with the generation of coherent and contextually relevant text based on the information retrieved. This model was specifically chosen for its adeptness at understanding nuanced language and generating detailed responses. Its integration signifies the system's ability to not just find but also intelligently expand upon the information retrieved, creating comprehensive answers.

- **Chunking and Indexing Strategy**: Prior to indexing, the dataset undergoes a meticulous chunking process. This step is designed to break down articles into manageable, information-dense fragments. Such a strategy ensures that the retrieval engine can work efficiently, focusing on chunks that are rich in content yet concise enough to maintain high retrieval speeds. Each chunk is then embedded and indexed using FAISS, facilitating rapid and accurate search capabilities.

## Technology Selection and Limitations
The construction of the system was guided by a careful selection of libraries and tools, chosen for their robustness, efficiency, and compatibility with the project's requirements. Here's a deeper dive into the technologies employed:

- **FAISS (Facebook AI Similarity Search)**: Utilized for its exceptional speed and efficiency in conducting similarity searches and clustering dense vectors, FAISS is a cornerstone of the retrieval engine. Its ability to handle large-scale datasets with minimal latency makes it an ideal choice for indexing and retrieving text fragments from the corpus. FAISS's flexibility in supporting different types of indexes also allows for fine-tuning based on the specific needs of the dataset and retrieval goals.

- **Transformers Library by Hugging Face**: The generative model's backbone is the Transformers library, which provides access to thousands of pre-trained models tailored for a variety of NLP tasks. Specifically, the "phi-1_5" model from this library was chosen for its advanced language comprehension and generation capabilities. The Transformers library also facilitates the integration of these models into the system, with extensive support for model manipulation and deployment.

- **Sentence Transformers**: For generating sentence embeddings, the system leverages the Sentence Transformers library. This library, an extension of the Transformers library, is optimized for creating sentence-level embeddings. The "all-MiniLM-L6-v2" model, known for its efficiency and accuracy in capturing semantic meanings, was selected. Sentence Transformers significantly reduce the effort and complexity involved in embedding generation, offering a streamlined approach for converting text data into a format suitable for similarity search and retrieval.

- **PyTorch**: Underlying both the FAISS library and the Transformers-based models is PyTorch, a flexible and powerful deep learning framework. PyTorch's dynamic computation graph paradigm enables intuitive model development and easier debugging, a critical advantage during the development of complex NLP systems. Its comprehensive ecosystem and active community support were instrumental in implementing the system's machine learning components.

- **Limitations Due to Lack of CUDA Support**: A notable constraint in the development process was the non-availability of CUDA, which restricted the system to CPU-based computations. This limitation impacted the selection of models and computational strategies, steering the project towards technologies that maintain efficiency on CPU environments. The constraint necessitated optimizations such as thread management in PyTorch and the selection of models with a lower computational footprint, ensuring that the system remained functional and responsive without GPU acceleration.

## Challenges Encountered

Throughout the system's development, several challenges were confronted, not least of which stemmed from technical, performance, and learning aspects:

- **First-Time Package Development**: One of the foundational challenges was the creation of a Python package for the first time. This endeavor required a deep dive into Python's packaging conventions, module structuring, and distribution mechanisms. Learning to navigate these complexities while ensuring the package's usability and maintainability presented a significant learning curve.

- **Data Preprocessing**: Ensuring the input data's cleanliness and relevance was paramount. Developing a robust preprocessing pipeline, capable of efficiently cleaning, tokenizing, and preparing text data for both the retrieval and generative components, required extensive testing and refinement.

- **Model Integration**: The integration of disparate models—FAISS for retrieval and the PHI "phi-1_5" model for generation—into a cohesive system architecture was complex. It necessitated a thoughtful approach to data flow and interaction between components, ensuring that the retrieved information was effectively utilized by the generative model.

- **Performance Optimization for CPU**: Given the system's development and testing on a platform without CUDA support, optimizing for CPU performance was a priority. This challenge was addressed through various strategies, including optimizing the number of threads for processing and selecting models that offer a balance between performance and computational demand.


## Future Enhancement

As the system continues to evolve, several key areas have been identified for enhancement to elevate its performance, usability, and scope of application:

- **Model Upgrade with CUDA Support**: With the current system optimized for CPU performance due to hardware limitations, a significant future enhancement involves leveraging CUDA capabilities, should they become available. Access to CUDA would enable the integration of more advanced generative models that utilize attention mechanisms extensively, such as larger variants of the Phi model or other state-of-the-art models. These models can potentially offer improved text generation quality due to their ability to process and integrate a broader context more effectively.

- **User Interface Development**: To enhance accessibility and ease of use, the development of a graphical user interface (GUI) or an application programming interface (API) is planned. This would make the system more approachable for a wider range of users, facilitating easier interaction with the system's capabilities.

- **Continuous Refinement and Testing**: Ongoing efforts to tune the retrieval and generation models will focus on improving the accuracy and relevance of responses. Additionally, systematic testing will be conducted to optimize the system's performance, especially in response latency and handling of large datasets.

- **Scalability and Domain Adaptation**: Future development efforts will also explore scaling the system to accommodate larger datasets more efficiently and adapting it for specific knowledge domains or languages. This includes investigating techniques for domain-specific model fine-tuning and exploring multi-lingual support.

- **Exploring Attention Mechanisms**: With the potential upgrade to hardware that supports CUDA, exploring attention mechanisms more deeply will be a priority. Attention-based models offer nuanced understanding and generation capabilities, particularly for complex queries that require a detailed synthesis of multiple information sources.


## Conclusion

The developed RAG system employs an indexing and retrieval strategy tailored for efficient information searchability, paired with a generative model that produces detailed responses based on the retrieved contexts. While the current iteration effectively achieves the fundamental goals set for the system, ongoing development aims to refine these capabilities further.