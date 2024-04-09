import spacy
import swifter
import nltk
from nltk.corpus import stopwords

nltk.download('popular')
spacy.cli.download("en_core_web_sm")

class Preprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def preprocess_text(self, text):
        # Process the text through spaCy NLP pipeline
        doc = self.nlp(text)
        processed_tokens = []
        
        for token in doc:
            # Preserve named entities as they are
            if token.ent_type_:
                processed_tokens.append(token.text)
            # Preserve nouns and certain POS tags, exclude stopwords and punctuation
            elif token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop and not token.is_punct:
                processed_tokens.append(token.text)
            # Apply lemmatization and lowercasing to other tokens
            else:
                if not token.is_stop and not token.is_punct:
                    processed_tokens.append(token.lemma_.lower())
        
        return ' '.join(processed_tokens)


    def chunk_article(self, text, max_chunk_length=512):
        """
        Splits the article into manageable chunks, each not exceeding the specified maximum length.
        
        Args:
        text (str): The article text to be chunked.
        max_chunk_length (int): The maximum allowed length of each chunk.
        
        Returns:
        list: A list of text chunks.
        """
        # Split the article into paragraphs
        paragraphs = text.split('\n\n')
        
        # Further split into sentences if needed, based on the heuristic like length
        chunks = []
        for paragraph in paragraphs:
            sentences = nltk.sent_tokenize(paragraph)
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # Handle the case where a single sentence is longer than the max chunk length
                if sentence_length > max_chunk_length:
                    if current_chunk:  # If the current chunk is not empty, add it to the chunks list
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    # Here you could further split the sentence or truncate it to fit the max length
                    # For simplicity, we'll add the long sentence as its own chunk
                    chunks.append(sentence)
                    continue
                
                if current_length + sentence_length > max_chunk_length:
                    # If this sentence would exceed the max length, add the current chunk first
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            # After processing all sentences in a paragraph, add the remaining current chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        return chunks
