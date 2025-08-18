import spacy
from collections import Counter
from nltk.stem import PorterStemmer

# Load English model
nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

# Sample text
text = "Studies show that the quick brown foxes jumped over the lazy dogs. Running fast is good! The quixotic idea was deemed too idiosyncratic."

def nlp_preprocessing(text):
    doc = nlp(text)
    
    # Step 1: Tokenization + Count word frequencies for rare word removal
    word_freq = Counter(token.text.lower() for token in doc if not token.is_punct)
    
    print("Original Text:", text)
    print("\n=== Preprocessing Steps ===\n")
    
    # 1. Tokenization
    tokens = [token.text for token in doc]
    print("1. Tokenization:", tokens)
    
    # 2. Punctuation Removal
    tokens_no_punct = [token.text for token in doc if not token.is_punct]
    print("\n2. Punctuation Removal:", tokens_no_punct)
    
    # 3. Normalization (Lowercasing)
    normalized_tokens = [token.text.lower() for token in doc if not token.is_punct]
    print("\n3. Normalization (Lowercase):", normalized_tokens)
    
    # 4. Stop Word Removal
    tokens_no_stop = [token.text for token in doc if not token.is_stop and not token.is_punct]
    print("\n4. Stop Word Removal:", tokens_no_stop)
    
    # 5. Rare Word Removal (threshold = 1 occurrence)
    rare_words = {word for word, count in word_freq.items() if count <= 1}
    tokens_no_rare = [token.text for token in doc 
                     if not token.is_punct and word_freq[token.text.lower()] > 1]
    print("\n5. Rare Word Removal (frequency ≤ 1):", tokens_no_rare)
    print("   Removed words:", rare_words)
    
    # 6. Lemmatization
    lemmas = [token.lemma_ for token in doc if not token.is_punct]
    print("\n6. Lemmatization:", lemmas)
    
    # 7. Word Stemming (using NLTK's Porter Stemmer)
    stems = [stemmer.stem(token.text) for token in doc if not token.is_punct]
    print("\n7. Word Stemming (Porter):", stems)
    
    # Final Preprocessed Output
    processed_tokens = [
        stemmer.stem(token.lemma_.lower()) 
        for token in doc 
        if (not token.is_punct and 
            not token.is_stop and 
            word_freq[token.text.lower()] > 1)
    ]
    print("\n=== Final Preprocessed Output ===")
    print(processed_tokens)

# Run preprocessing
nlp_preprocessing(text)