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
    # document1 = st.file_uploader('Upload Document 1')
with col2:
    st.markdown('<p style="font-size:20px; font-weight:400; color:#4E4E61; margin-bottom:12px">Document 2</p>', unsafe_allow_html=True)
    text2 = st.text_area('', height=300, placeholder="Type your text...", label_visibility="collapsed", key='doc2')
    # document2 = st.file_uploader('Upload Document 2')

st.markdown('<p style="margin-bottom:32px"></p>', unsafe_allow_html=True)
if st.button('Check Similarity'):
    st.write('Result: Similar')
    st.write('Confidence: 80%')
