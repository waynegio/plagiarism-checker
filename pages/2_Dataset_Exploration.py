import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
from sklearn.preprocessing import MinMaxScaler
import os
import pickle

train_df = pd.read_excel('./MSRParaphraseCorpus/msr_paraphrase_train.xlsx')
test_df = pd.read_excel('./MSRParaphraseCorpus/msr_paraphrase_test.xlsx')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@100..900&display=swap');

html, body, h1, h2, h3, p, div, span {
    font-family: 'Lexend', sans-serif;
}

h3 {
    margin-top: 10px;
}
            
.stMarkdown h1 a,
.stMarkdown h2 a,
.stMarkdown h3 a,
.stMarkdown h4 a {
    display: none !important;
}

[data-testid="stHeader"] {
    display: none;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        #DEDEE6 0%,
        #FEFEFF 40%
    );
}

.block-container {
    padding-top: 80px;
    padding-bottom: 160px;
    padding-left: 120px;
    padding-right: 120px;
    max-width: 100%;
}

.card {
    padding: 20px;
    border-radius: 20px;
    background-color: #FEFEFF;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    height: 130px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    justify-content: center;
    color: #4E4E61;
}
            
.info {
    border-radius: 20px;
    background-color: #7F7FA4;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    color: #FEFEFF;
    height: 56px;
    padding: 24px;
    display: flex;
    align-items: center;
}

.section {
    margin-top: 40px;
}

.title {
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 32px;
    color: #4E4E61;
}

.subtitle {
    font-size: 20px;
    font-weight: 500;
    color: #4E4E61;
    margin-bottom: 4px;
}

.subtitle2{
    font-size: 16px;
    font-weight: 500;
    color: #4E4E61;
    margin-bottom: 8px;
}

.text {
    font-size: 16px;
    font-weight: 300;
    color: #4E4E61;
    margin-bottom: 32px;
}
            
[data-testid="stButton"] button {
    background-color: #7F7FA4;
    color: #FEFEFF;
    border: 1px solid #7F7FA4;
    border-radius: 32px;
    width: 210px;
    height: 55px;
}

[data-testid="stVerticalBlock"] {
    align-items: end;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title='Dataset Exploration', page_icon='📊')

st.markdown('<div class="title">📊 Dataset Exploration</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="card">
        <p style="font-size:14px; color:#7E7E84; margin:0;">Total Data</p>
        <p style="font-size:32px; margin:5px 0 0 0;">{len(train_df)}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <p style="font-size:14px; color:#7E7E84; margin:0;">Columns</p>
        <p style="font-size:32px; margin:5px 0 0 0;">{len(train_df.columns)}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    missing = train_df.isnull().sum().sum()
    st.markdown(f"""
    <div class="card">
        <p style="font-size:14px; color:#7E7E84; margin:0;">Missing Values</p>
        <p style="font-size:32px; margin:5px 0 0 0;">{missing}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section"></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle" style="margin-bottom:24px">Data Quality Check</div>', unsafe_allow_html=True)

col4, col5 = st.columns(2)

with col4:
    st.markdown(f"""<div class="info">Duplicate Rows: {train_df.duplicated().sum()}</div>""", unsafe_allow_html=True)

with col5:
    st.markdown(f"""<div class="info">Duplicate Columns: {train_df.columns.duplicated().sum()}</div>""", unsafe_allow_html=True)

st.markdown('<div class="section"></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle" style="margin-bottom:24px">Dataset Preview</div>', unsafe_allow_html=True)

st.dataframe(train_df, width='stretch')

st.markdown('<div class="section"></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle" style="margin-bottom:24px">Label Distribution</div>', unsafe_allow_html=True)

label_counts = train_df["Quality"].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))

if 0 in label_counts.index:
    bar0 = ax.bar("0", label_counts[0], color="#CFCEF9", label='0 = Non-Similar')
    ax.bar_label(bar0, padding=3)

if 1 in label_counts.index:
    bar1 = ax.bar("1", label_counts[1], color="#C0D2F6", label='1 = Similar')
    ax.bar_label(bar1, padding=3)

ax.set_xlabel("Label")
ax.set_ylabel("Count")
ax.legend()
ax.set_ylim(0, label_counts.max() * 1.1)

st.pyplot(fig)

st.markdown('<div class="text" style="margin:0">'
'The dataset contains 1,320 samples in class "Non-Similar" and 2,751 samples in class "Similar", resulting in an approximate ratio of 1:2.1, which indicates a moderate class imbalance. This level of imbalance is not considered severe, so additional balancing techniques may not be immediately necessary. However, class imbalance can still make the model more biased toward the majority class, which may reduce the model\'s ability to correctly identify the minority class. Therefore, relying only on accuracy is not recommended. More suitable evaluation metrics include precision, recall, F1-score, and macro F1-score, as these provide a better understanding of performance across both classes.'
'</div>', unsafe_allow_html=True)

st.markdown('<div class="section"></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle" style="margin-bottom:24px">Data Preprocessing</div>', unsafe_allow_html=True)

preprocess_data = {
    'Similarity Features': ['TF-IDF Cosine Similarity', 'Jaccard Similarity', 'Levenshtein Distance', '3-gram Similarity', 'S-BERT'],
    'Lowercase'         : ['☑️', '☑️', '☑️', '☑️', '☑️'],
    'Remove Punctuation': ['☑️', '☑️', '☑️', '☑️', '✖️'],
    'Remove Stopwords'  : ['☑️', '☑️', '✖️', '✖️', '✖️'],
    'Lemmatize'         : ['☑️', '☑️', '✖️', '✖️', '✖️'],
}

preprocess_df = pd.DataFrame(preprocess_data)
st.dataframe(preprocess_df, hide_index=True)

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

st.markdown('<div class="section"></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle" style="margin-bottom:24px">Feature Engineering</div>', unsafe_allow_html=True)

st.markdown('<div class="text">The five similarity features were selected to capture different aspects of text similarity and reduce feature redundancy by combining lexical, structural, and semantic perspectives. TF-IDF Cosine Similarity and Jaccard Similarity were chosen to detect direct copying or texts with similar vocabularies. Levenshtein Distance was included to identify small edits or minor modifications. 3-gram Similarity was used to detect partial copying or reordered text segments. Finally, S-BERT was selected to detect paraphrased plagiarism by identifying texts with similar meanings even when different words are used.</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle2">1. Cosine Similarity with TF-IDF</div>', unsafe_allow_html=True)
st.markdown('<div class="text">Cosine Similarity with TF-IDF measures the similarity between two documents by representing each text as a weighted vector of terms, where important words receive higher weights while common words receive lower importance. In plagiarism detection, this method is useful for detecting direct copying or texts with only minor word modifications.</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle2">2. Jaccard Similarity</div>', unsafe_allow_html=True)
st.markdown('<div class="text">Jaccard Similarity calculates the proportion of shared unique words between two documents by comparing the intersection and union of their token sets. In plagiarism detection, it is commonly used to measure word overlap and can effectively identify texts that share many identical terms.</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle2">3. Levenshtein Distance</div>', unsafe_allow_html=True)
st.markdown('<div class="text">Levenshtein Distance measures the minimum number of insertions, deletions, or substitutions required to transform one text into another. In plagiarism detection, this method is useful for detecting small edits, typographical changes, or lightly modified copied content.</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle2">4. 3-gram Similarity</div>', unsafe_allow_html=True)
st.markdown('<div class="text">3-gram Similarity compares documents based on overlapping sequences of three consecutive characters or words, allowing it to capture local phrase patterns and partial sequence matches. In plagiarism detection, this method is useful for identifying copied phrases, reordered text segments, or partial duplication that may not be fully captured by unigram-based methods.</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle2">5. S-BERT (Sentence-BERT)</div>', unsafe_allow_html=True)
st.markdown('<div class="text">S-BERT converts sentences or documents into dense semantic embeddings that capture contextual meaning rather than relying solely on exact word overlap. In plagiarism detection, this method is highly effective for identifying paraphrased plagiarism, as it can detect texts with similar meanings even when different vocabulary or sentence structures are used.</div>', unsafe_allow_html=True)

text1_col = "#1 String"
text2_col = "#2 String"

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

scaler = MinMaxScaler()

if os.path.exists("MSRParaphraseCorpus/msr_paraphrase_train_features.csv"):
    train_feature_df = pd.read_csv("MSRParaphraseCorpus/msr_paraphrase_train_features.csv")
else:
    features = []

    for _, row in train_df.iterrows():
        text1 = str(row[text1_col])
        text2 = str(row[text2_col])

        features.append({
            "tfidf_cosine": tfidf_cosine(preprocess_cosine_jaccard(text1), preprocess_cosine_jaccard(text2)),
            "jaccard": jaccard_similarity(preprocess_cosine_jaccard(text1), preprocess_cosine_jaccard(text2)),
            "levenshtein": levenshtein_similarity(preprocess_levenshtein_trigram(text1), preprocess_levenshtein_trigram(text2)),
            "trigram_similarity": trigram_similarity(preprocess_levenshtein_trigram(text1), preprocess_levenshtein_trigram(text2)),
            "sbert_similarity": sbert_similarity(preprocess_sbert(text1), preprocess_sbert(text2))
        })

    train_feature_df = pd.DataFrame(features)
    train_feature_df = pd.DataFrame(scaler.fit_transform(train_feature_df), columns=train_feature_df.columns)
    train_feature_df.to_csv("MSRParaphraseCorpus/msr_paraphrase_train_features.csv", index=False)

feature_df = pd.concat([train_df['Quality'], train_feature_df], axis=1)
st.dataframe(feature_df.head(10))

st.markdown('<div class="section"></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle" style="margin-bottom:24px">Feature Correlation</div>', unsafe_allow_html=True)

corr = feature_df.corr()
label_corr = corr['Quality'].drop('Quality')
st.dataframe(label_corr)

st.markdown('<div class="text">The correlation analysis shows that most features have moderate relationships with the target label (Quality), indicating that they provide useful information for distinguishing paraphrase or plagiarism pairs. Among all features, S-BERT Similarity has the highest correlation with the label (0.42), followed by Levenshtein Distance (0.39), while TF-IDF Cosine Similarity and Jaccard Similarity both show similar correlations of around 0.38. These values are not considered too low for text classification tasks, as similarity-based features often have moderate rather than extremely high correlations due to the complexity of language patterns.</div>', unsafe_allow_html=True)

if os.path.exists("MSRParaphraseCorpus/msr_paraphrase_test_features.csv"):
    test_feature_df = pd.read_csv("MSRParaphraseCorpus/msr_paraphrase_test_features.csv")
else:
    features = []

    for _, row in test_df.iterrows():
        text1 = str(row[text1_col])
        text2 = str(row[text2_col])

        features.append({
            "tfidf_cosine": tfidf_cosine(preprocess_cosine_jaccard(text1), preprocess_cosine_jaccard(text2)),
            "jaccard": jaccard_similarity(preprocess_cosine_jaccard(text1), preprocess_cosine_jaccard(text2)),
            "levenshtein": levenshtein_similarity(preprocess_levenshtein_trigram(text1), preprocess_levenshtein_trigram(text2)),
            "trigram_similarity": trigram_similarity(preprocess_levenshtein_trigram(text1), preprocess_levenshtein_trigram(text2)),
            "sbert_similarity": sbert_similarity(preprocess_sbert(text1), preprocess_sbert(text2))
        })

    test_feature_df = pd.DataFrame((features))
    test_feature_df = pd.DataFrame(scaler.transform(test_feature_df), columns=test_feature_df.columns)
    test_feature_df.to_csv("MSRParaphraseCorpus/msr_paraphrase_test_features.csv", index=False)

with open("model/scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

if st.button("Let's Build The Model"):
    st.switch_page("pages/3_Build_The_Model.py")