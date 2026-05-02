
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein
from sentence_transformers import SentenceTransformer, util
from nltk.util import ngrams
import pandas as pd

lemmatizer = WordNetLemmatizer()
punctuations = string.punctuation
eng_stopwords = set(stopwords.words('english'))

def get_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('R'):
        return 'r'
    elif tag.startswith('V'):
        return 'v'
    else:
        return 'n'

def lemmatizing(words):
    tagged = pos_tag(words)
    lemmatized = [lemmatizer.lemmatize(word, get_pos(tag)) for word, tag in tagged]
    return lemmatized

def preprocess_cosine_jaccard(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in punctuations and token not in eng_stopwords and not token.isnumeric()]
    tokens = lemmatizing(tokens)
    return ' '.join(tokens)

def preprocess_levenshtein_trigram(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def preprocess_sbert(text):
    return text.lower().strip()

def tfidf_cosine(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return score

def jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union == 0:
        return 0
    
    return intersection / union

def levenshtein_similarity(text1, text2):
    return Levenshtein.ratio(text1, text2)

def trigram_similarity(text1, text2):
    grams1 = set(ngrams(text1.split(), 3))
    grams2 = set(ngrams(text2.split(), 3))

    intersection = len(grams1.intersection(grams2))
    union = len(grams1.union(grams2))

    return intersection / union if union != 0 else 0

model = SentenceTransformer('all-MiniLM-L6-v2')

def sbert_similarity(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)

    score = util.cos_sim(emb1, emb2).item()
    return score

def extract_features(text1, text2):
    features = {
        "tfidf_cosine": tfidf_cosine(
            preprocess_cosine_jaccard(text1),
            preprocess_cosine_jaccard(text2)
        ),
        "jaccard": jaccard_similarity(
            preprocess_cosine_jaccard(text1),
            preprocess_cosine_jaccard(text2)
        ),
        "levenshtein": levenshtein_similarity(
            preprocess_levenshtein_trigram(text1),
            preprocess_levenshtein_trigram(text2)
        ),
        "trigram_similarity": trigram_similarity(
            preprocess_levenshtein_trigram(text1),
            preprocess_levenshtein_trigram(text2)
        ),
        "sbert_similarity": sbert_similarity(
            preprocess_sbert(text1),
            preprocess_sbert(text2)
        )
    }

    feature_df = pd.DataFrame([features])
    similarity_score = feature_df.mean(axis=1).iloc[0]

    return feature_df, similarity_score