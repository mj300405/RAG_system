import spacy
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK models and corpora
nltk.download('popular')
# Ensure the spaCy English model is downloaded
spacy.cli.download("en_core_web_sm")

class Preprocessor:
    """
    Preprocessor class for cleaning and chunking text data using spaCy and NLTK.
    
    This class provides methods for preprocessing text by applying tokenization,
    named entity recognition, part-of-speech tagging, lemmatization, and chunking
    strategies to prepare text data for NLP tasks.
    
    Attributes:
        nlp (spacy.lang): Loaded spaCy language model for English.
    """
    
    def __init__(self):
        """
        Initializes the Preprocessor class by loading the spaCy English model.
        """
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess_text(self, text):
        """
        Processes and cleans a given text by applying various NLP techniques.
        
        Named entities are preserved as is. Nouns and certain part-of-speech tags
        are also preserved, excluding stopwords and punctuation. Other tokens are
        lemmatized and converted to lowercase.
        
        Args:
            text (str): The text to preprocess.
        
        Returns:
            str: A cleaned and processed string.
        """
        doc = self.nlp(text)
        processed_tokens = []
        
        for token in doc:
            # Preserve named entities as they are
            if token.ent_type_:
                processed_tokens.append(token.text)
            # Preserve nouns, proper nouns, excluding stopwords and punctuation
            elif token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and not token.is_punct:
                processed_tokens.append(token.text)
            # Lemmatize and lowercase other tokens, excluding stopwords and punctuation
            else:
                if not token.is_stop and not token.is_punct:
                    processed_tokens.append(token.lemma_.lower())
        
        return ' '.join(processed_tokens)

    def chunk_article(self, text, max_chunk_length=512):
        """
        Splits the text into manageable chunks, each not exceeding the specified maximum length,
        aiming for coherent segments by respecting sentence boundaries.
        
        Args:
            text (str): The article text to be chunked.
            max_chunk_length (int): The maximum allowed length of each chunk.
        
        Returns:
            list: A list of text chunks, each under the maximum length.
        """
        paragraphs = text.split('\n\n')
        
        chunks = []
        for paragraph in paragraphs:
            sentences = nltk.sent_tokenize(paragraph)
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # Check if adding the sentence exceeds the max chunk length
                if current_length + sentence_length > max_chunk_length:
                    # Add the current chunk to chunks list and start a new chunk
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    # Add the sentence to the current chunk
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            # Add any remaining sentences in the current chunk to the chunks list
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        return chunks
