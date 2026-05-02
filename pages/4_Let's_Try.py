import pickle
import os
import pandas as pd
import streamlit as st
from preprocessing import (
    preprocess_cosine_jaccard,
    preprocess_levenshtein_trigram,
    preprocess_sbert,
    tfidf_cosine,
    jaccard_similarity,
    levenshtein_similarity,
    trigram_similarity,
    sbert_similarity,
    extract_features
)

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
        #FEFEFF 30%
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
    align-items: center;
    color: #4E4E61;
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
    align-items: center;
}

[data-testid="stSpinner"] {
    color: #7F7FA4;
}

div[data-testid="stTextArea"] > div {
    background-color: #FEFEFF !important;
    border-radius: 32px !important;
    border: 1px solid #C7C7D2 !important;
}

textarea {
    background-color: #FEFEFF !important;
    color: #4E4E61 !important;
    border: none !important;
    padding: 24px !important;
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title='Similarity Check', page_icon='📋')

st.markdown('<p style="font-size:32px; font-weight:400; color:#4E4E61; margin-bottom:48px">Check Document Similarity</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<p style="font-size:20px; font-weight:400; color:#4E4E61; margin-bottom:12px">Document 1</p>', unsafe_allow_html=True)
    text1 = st.text_area('', height=300, placeholder="Type your text...", label_visibility="collapsed", key='doc1')
with col2:
    st.markdown('<p style="font-size:20px; font-weight:400; color:#4E4E61; margin-bottom:12px">Document 2</p>', unsafe_allow_html=True)
    text2 = st.text_area('', height=300, placeholder="Type your text...", label_visibility="collapsed", key='doc2')

st.markdown('<p style="margin-bottom:40px"></p>', unsafe_allow_html=True)

if st.button('Check Similarity'):
    if not text1.strip() or not text2.strip():
        st.warning('Please enter text in both fields.')
    else:
        st.markdown('<p style="margin-bottom:24px"></p>', unsafe_allow_html=True)
        with st.spinner('Analyzing similarity...'):
            model_path = "model/logistic_model.pkl"
            scaler_path = "model/scaler.pkl"

            with open(model_path, "rb") as file:
                model = pickle.load(file)

            with open(scaler_path, "rb") as file:
                scaler = pickle.load(file)

            features, similarity_score = extract_features(text1, text2)
            
            feature_names = features.columns
            feature_scaled = scaler.transform(features)
            feature_scaled_df = pd.DataFrame(feature_scaled, columns=feature_names)

            prediction = model.predict(feature_scaled)[0]
            probability = model.predict_proba(feature_scaled)[0].max()

        st.markdown(f'''
        <div style="
            background: #FEFEFF;
            border: 1px solid #C7C7D2;
            border-radius: 24px;
            padding: 32px;
            margin-top: 8px;
            max-width: 480px;
            margin-left: auto;
            margin-right: auto;
            text-align: center;
        ">
            <p style="font-size:16px; color:#4E4E61; margin-bottom:16px">
                Similarity Score
            </p>
            <p style="font-size:56px; font-weight:600; color:#7F7FA4; margin:0">
                {similarity_score:.2%}
            </p>
            <p style="font-size:20px; color:#7F7FA4; margin-top:8px">
                {"Paraphrase Detected" if similarity_score >= 0.5 else "Not A Paraphrase"}
            </p>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown('<div class="section"></div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle" style="margin-bottom:24px">Similarity Features</div>', unsafe_allow_html=True)

        st.dataframe(feature_scaled_df, hide_index=True)

        st.markdown(f"""
        <div class="card" style="margin-top:32px">
            <p style="font-size:14px; color:#7E7E84; margin:0;">Model Confidence</p>
            <p style="font-size:32px; margin:5px 0 0 0;">{probability:.2%}</p>
        </div>
        """, unsafe_allow_html=True)