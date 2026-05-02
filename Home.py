import streamlit as st
import nltk

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

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
    padding-top: 160px;
    padding-bottom: 160px;
    padding-left: 120px;
    padding-right: 120px;
    margin: 0;
    max-width: 100%;
}

[data-testid="stButton"] button {
    background-color: transparent;
    color: #4E4E61;
    border: 1px solid #7F7FA4;
    border-radius: 32px;
    width: 210px;
    height: 55px;
}

[data-testid="stVerticalBlock"] {
    align-items: center;
}

.stApp {
    background-color: #DEDEE6 !important;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title='Home', page_icon='🔎')

st.markdown('<p style="text-align:center; font-size:24px; font-weight:300; color:#4E4E61; margin-bottom:12px">P A I R C H E C K</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-size:64px; font-weight:500; color:#4E4E61">Pairwise Document</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-size:64px; font-weight:500; color:#4E4E61; margin-top:-24px">Similarity Checker</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-size:16px; font-weight:300; color:#8D8D93; margin-top:24px">A machine learning-based tool for pairwise document comparison,</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; font-size:16px; font-weight:300; color:#8D8D93; margin-top:-4px; margin-bottom:56px">designed to evaluate similarity and plagiarism risk.</p>', unsafe_allow_html=True)

_, col1, col2, _ = st.columns([1, 2, 2, 1])
with col1:
    if st.button("How It Works"):
        st.switch_page("pages/1_How_It_Works.py")
with col2:
    if st.button("Let's Try"):
        st.switch_page("pages/4_Let's_Try.py")