import streamlit as st
st.set_page_config(page_title='Dataset Exploration', page_icon='📊')
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('./MSRParaphraseCorpus/msr_paraphrase_train.xlsx')

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
}

.section {
    margin-top: 40px;
}

.title {
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📊 Dataset Exploration</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="card">
        <p style="font-size:14px; color:#6b7280; margin:0;">Total Data</p>
        <h2 style="margin:5px 0 0 0;">{len(df)}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <p style="font-size:14px; color:#6b7280; margin:0;">Columns</p>
        <h2 style="margin:5px 0 0 0;">{len(df.columns)}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    missing = df.isnull().sum().sum()
    st.markdown(f"""
    <div class="card">
        <p style="font-size:14px; color:#6b7280; margin:0;">Missing Values</p>
        <h2 style="margin:5px 0 0 0;">{missing}</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section"></div>', unsafe_allow_html=True)
st.markdown("### Data Quality Check")

col4, col5 = st.columns(2)

with col4:
    st.info(f"Duplicate Rows: {df.duplicated().sum()}")

with col5:
    st.info(f"Duplicate Columns: {df.columns.duplicated().sum()}")

st.markdown('<div class="section"></div>', unsafe_allow_html=True)
st.markdown("### Dataset Preview")
st.dataframe(df, width='stretch')

st.markdown('<div class="section"></div>', unsafe_allow_html=True)
st.markdown("### Label Distribution")

label_counts = df["Quality"].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(label_counts.index.astype(str), label_counts.values)
ax.set_xlabel("Label")
ax.set_ylabel("Count")
ax.set_title("Paraphrase Distribution")

st.pyplot(fig)