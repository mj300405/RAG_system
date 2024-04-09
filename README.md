# RAG System

Welcome to RAG system, a tool designed to retrieve articles based on your queries efficiently.

## Prerequisites

Before installing and using the retrieval system, make sure you have Python 3.6 or higher installed on your system. You can download Python from [here](https://www.python.org/downloads/).

## Installation

Clone the repository and navigate into it:

```bash
git clone https://github.com/mj300405/RAG_system.git
cd RAG_system
```

It's recommended to use a virtual environment to keep dependencies required by different projects separate by creating isolated Python virtual environments. Here's how you can do it on Windows, Linux, and macOS:

### Windows

```powershell
py -m venv venv
.\venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

After activating the virtual environment, install the package:

```bash
pip install .
```

## Initial Setup

Before using the retrieval system, you need to perform an initial setup. This setup involves preparing the dataset and indexing embeddings, which is done through scripts provided in the package. 

### Preparing the Dataset and Indexing Embeddings

This script processes the raw CSV data, chunks articles, generates embeddings, and prepares the dataset for retrieval, as well as indexes the embeddings for efficient retrieval.

```bash
prepare_dataset
```

## Usage

With the initial setup complete, you can now use the retrieval system to find articles related to your query. Run the following command and follow the prompts:

```bash
inference
```

## Additional Notes

- If you encounter any issues during the installation or setup process, ensure that your virtual environment is active and that you have installed all the necessary dependencies as listed in `requirements.txt`.
- To deactivate the virtual environment after use, simply run `deactivate` (Windows) or `source deactivate` (Linux/macOS).

Thank you for using RAG_system. For any issues or contributions, please feel free to open an issue or pull request on GitHub.
