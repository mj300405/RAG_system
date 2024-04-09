from setuptools import setup, find_packages

setup(
    name='RAG_system',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'nltk',
        'spacy',
        'swifter',
        'faiss-cpu',  # Note: Use 'faiss-gpu' if targeting GPU installations
        'sentence-transformers',
        'transformers',
        'datasets',
    ],
    entry_points={
        'console_scripts': [
            'inference=RAG_system.inference:main',
            'prepare-dataset=scripts.prepare_dataset:prepare_dataset'
        ],
    },
)
