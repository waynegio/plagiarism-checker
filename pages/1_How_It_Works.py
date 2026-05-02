import streamlit as st

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@100..900&display=swap');

html, body, h1, h2, h3, p, div, span {
    font-family: 'Lexend', sans-serif;
    margin: 0;
    padding: 0;
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
    margin: 0;
    max-width: 100%;
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
    margin-bottom: 8px;
}

.text {
    font-size: 16px;
    font-weight: 300;
    color: #7E7E84;
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

.stApp, [data-testid="stAppViewContainer"] {
    background: #DEDEE6 !important;
}

p, h1, h2, h3, div {
    color: #4E4E61 !important;
}

[data-testid="stButton"] button {
    color: #FEFEFF !important;
    background-color: #7F7FA4 !important;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title='How It Works', page_icon='❓')

st.markdown('<div class="title">❓ How PairCheck Works?</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">1. Upload Documents</div>', unsafe_allow_html=True)
st.markdown('<div class="text">Upload two documents that you want to compare or paste text directly. The system accepts text input or supported document files for analysis.</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">2. Text Preprocessing</div>', unsafe_allow_html=True)
st.markdown('<div class="text">Both documents are cleaned and normalized through lowercase conversion, punctuation removal, stopwords filtering, lemmatizing, and MinMax normalization. This helps improve consistency before similarity analysis.</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">3. Feature Extraction</div>', unsafe_allow_html=True)
st.markdown('<div class="text">The system extracts multiple similarity features such as cosine similarity, Jaccard similarity, Levenshtein distance, n-gram overlap, and Sentence-BERT. These features represent how closely the two documents are related.</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">4. Similarity Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="text">Extracted similarity features are analyzed by a Logistic Regression model. The model determines whether the document pair is likely plagiarized or not.</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">5. Result and Confidence</div>', unsafe_allow_html=True)
st.markdown('<div class="text" style="margin-bottom:40px">The final prediction is displayed as similar or non-similar. A confidence score is also provided to indicate prediction certainty.</div>', unsafe_allow_html=True)

if st.button("Explore Data"):
    st.switch_page("pages/2_Dataset_Exploration.py")