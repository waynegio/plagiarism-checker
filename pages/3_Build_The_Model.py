import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix)
import pickle
import os

train_df = pd.read_excel('./MSRParaphraseCorpus/msr_paraphrase_train.xlsx')
train_features_df = pd.read_csv("./MSRParaphraseCorpus/msr_paraphrase_train_features.csv")
train_features_df = pd.concat([train_df['Quality'], train_features_df], axis=1)

test_df = pd.read_excel('./MSRParaphraseCorpus/msr_paraphrase_test.xlsx')
test_features_df = pd.read_csv("./MSRParaphraseCorpus/msr_paraphrase_test_features.csv")
test_features_df = pd.concat([test_df['Quality'], test_features_df], axis=1)

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

.stApp, [data-testid="stAppViewContainer"] {
    background: #DEDEE6 !important;
}

p, h1, h2, h3, div, span {
    color: #4E4E61 !important;
}

[data-testid="stButton"] button {
    color: #FEFEFF !important;
    background-color: #7F7FA4 !important;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title='Model Train and Evaluation', page_icon='🖥️')

st.markdown('<div class="title">🖥️ Model Train and Evaluation</div>', unsafe_allow_html=True)

model_path = "model/logistic_model.pkl"

X_train = train_features_df.drop(columns=["Quality"])
y_train = train_features_df["Quality"]
X_test = test_features_df.drop(columns=["Quality"])
y_test = test_features_df["Quality"]

if os.path.exists(model_path):
    with st.spinner("Loading model..."):
        with open(model_path, "rb") as file:
            model = pickle.load(file)

    st.success("Model loaded successfully!")

else:
    with st.spinner("Training and saving model..."):
        model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )

        model.fit(X_train, y_train)

        with open(model_path, "wb") as file:
            pickle.dump(model, file)

    st.success("Model trained and saved successfully!")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.markdown('<div class="section"></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle" style="margin-bottom:24px">Model Evaluation Metrics</div>', unsafe_allow_html=True)

st.markdown(f"""<div class="info">Classification Report:</div>""", unsafe_allow_html=True)
st.code(report, language="text")

st.markdown(f"""<div class="info">Confusion Matrix:</div>""", unsafe_allow_html=True)
cm_df = pd.DataFrame(
    cm,
    index=["Actual Negative", "Actual Positive"],
    columns=["Predicted Negative", "Predicted Positive"]
)

st.dataframe(cm_df, use_container_width=True)

if st.button("Let's Try"):
    st.switch_page("pages/4_Let's_Try.py")