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

st.set_page_config(page_title='How It Works', page_icon='❔')

st.markdown('<p style="font-size:32px; font-weight:400; color:#4E4E61">How PairCheck Works?</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:20px; font-weight:400; color:#4E4E61; margin-top:48px">1. Upload Documents</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:16px; font-weight:300; color:#7E7E84; margin-top:8px">Upload two documents that you want to compare or paste text directly. The system accepts text input or supported document files for analysis.</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:20px; font-weight:400; color:#4E4E61; margin-top:32px">2. Text Preprocessing</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:16px; font-weight:300; color:#7E7E84; margin-top:8px">Both documents are cleaned and normalized through lowercase conversion, punctuation removal, and stopwords filtering. This helps improve consistency before similarity analysis.</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:20px; font-weight:400; color:#4E4E61; margin-top:32px">3. Feature Extraction</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:16px; font-weight:300; color:#7E7E84; margin-top:8px">The system extracts multiple similarity features such as cosine similarity, Jaccard similarity, and n-gram overlap. These features represent how closely the two documents are related.</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:20px; font-weight:400; color:#4E4E61; margin-top:32px">4. Similarity Analysis</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:16px; font-weight:300; color:#7E7E84; margin-top:8px">Extracted similarity features are analyzed by a trained machine learning model. The model determines whether the document pair is likely plagiarized or not.</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:20px; font-weight:400; color:#4E4E61; margin-top:32px">5. Result and Confidence</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:16px; font-weight:300; color:#7E7E84; margin-top:8px; margin-bottom:40px">The final prediction is displayed as similar or non-similar. A confidence score is also provided to indicate prediction certainty.</p>', unsafe_allow_html=True)

if st.button("Explore Data"):
    st.switch_page("pages/2_Dataset_Exploration.py")