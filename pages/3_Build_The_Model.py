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
    color: #4E4E61;
}

/* ── Tooltips ── */
.tooltip-wrap {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.tooltip-wrap .stTooltip {
    font-size: 12px;
    color: #8888A0;
}

/* ── Center success message ── */
div[data-testid="stSuccess"] {
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
}

/* ── Fix native Streamlit tooltip overflow ── */
[data-testid="stTooltipIcon"] {
    white-space: normal !important;
    overflow: visible !important;
}
[data-testid="stTooltipContent"] {
    white-space: normal !important;
    word-wrap: break-word !important;
    max-width: 280px;
    line-height: 1.4;
}

/* ── Hover-only info icon tooltip ── */
.info-icon-wrap {
    display: inline-block;
    position: relative;
    cursor: default;
}
.info-icon-wrap .info-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #C7C7D2;
    color: #FEFEFF;
    font-size: 11px;
    font-weight: 700;
    font-style: normal;
    line-height: 1;
    margin-left: 6px;
    vertical-align: middle;
    cursor: help;
}
.info-icon-wrap .tooltip-text {
    visibility: hidden;
    opacity: 0;
    position: absolute;
    bottom: 130%;
    left: 50%;
    transform: translateX(-50%);
    background: #4E4E61;
    color: #FEFEFF;
    font-size: 12px;
    font-weight: 400;
    white-space: nowrap;
    padding: 6px 10px;
    border-radius: 8px;
    z-index: 999;
    pointer-events: none;
    transition: opacity 0.15s ease;
    font-family: 'Lexend', sans-serif;
}
.info-icon-wrap .tooltip-text::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 5px solid transparent;
    border-top-color: #4E4E61;
}
.info-icon-wrap:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
}

/* ── Tooltips ── */
.tooltip-wrap {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.tooltip-wrap .stTooltip {
    font-size: 12px;
    color: #8888A0;
}

/* ── Center success message ── */
div[data-testid="stSuccess"] {
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
}

/* ── Fix native Streamlit tooltip overflow ── */
[data-testid="stTooltipIcon"] {
    white-space: normal !important;
    overflow: visible !important;
}
[data-testid="stTooltipContent"] {
    white-space: normal !important;
    word-wrap: break-word !important;
    max-width: 280px;
    line-height: 1.4;
}

/* ── Hover-only info icon tooltip ── */
.info-icon-wrap {
    display: inline-block;
    position: relative;
    cursor: default;
}
.info-icon-wrap .info-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #C7C7D2;
    color: #FEFEFF;
    font-size: 11px;
    font-weight: 700;
    font-style: normal;
    line-height: 1;
    margin-left: 6px;
    vertical-align: middle;
    cursor: help;
}
.info-icon-wrap .tooltip-text {
    visibility: hidden;
    opacity: 0;
    position: absolute;
    bottom: 130%;
    left: 50%;
    transform: translateX(-50%);
    background: #4E4E61;
    color: #FEFEFF;
    font-size: 12px;
    font-weight: 400;
    white-space: nowrap;
    padding: 6px 10px;
    border-radius: 8px;
    z-index: 999;
    pointer-events: none;
    transition: opacity 0.15s ease;
    font-family: 'Lexend', sans-serif;
}
.info-icon-wrap .tooltip-text::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 5px solid transparent;
    border-top-color: #4E4E61;
}
.info-icon-wrap:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title='Model Train and Evaluation', page_icon='🖥️')

st.markdown('<div class="title">🖥️ Model Train and Evaluation</div>', unsafe_allow_html=True)

# ── Hyperparameter inputs ──────────────────────────────────────────────
def info_icon(label, description, min_val=None, max_val=None):
    tooltip = description
    if min_val is not None and max_val is not None:
        tooltip = f"{description}<br>Range: {min_val} - {max_val}"
    st.markdown(f"""
    <div style="display:flex;align-items:center;margin-bottom:4px;">
        <span style="font-size:14px;font-weight:500;">{label}</span>
        <div class="info-icon-wrap">
            <span class="info-icon">i</span>
            <span class="tooltip-text" style="background-color: white; color: rgb(127, 127, 164);">{tooltip}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


col_param1, col_param2 = st.columns(2)

with col_param1:
    info_icon("Regularization (C)", "Smaller = stronger regularisation", 0.01, 10.0)
    C = st.slider("", min_value=0.01, max_value=10.0, value=1.0, step=0.01, label_visibility="collapsed")

st.markdown('<p style="margin-bottom:12px"></p>', unsafe_allow_html=True)

col_param3, col_param4, col_param5 = st.columns(3)

with col_param3:
    info_icon("Solver", "Algorithm for optimisation")
    solver = st.selectbox("", ["lbfgs", "liblinear", "saga"], index=0, label_visibility="collapsed")

with col_param4:
    info_icon("Random State", "Seed for reproducibility")
    random_state = st.number_input("", value=42, step=1, label_visibility="collapsed")

with col_param5:
    info_icon("Max Iterations", "Max solver convergence iterations")
    max_iter = st.number_input("", value=1000, step=100, label_visibility="collapsed")

st.markdown("---")

X_train = train_features_df.drop(columns=["Quality"])
y_train = train_features_df["Quality"]
X_test = test_features_df.drop(columns=["Quality"])
y_test = test_features_df["Quality"]

model_path = "model/logistic_model.pkl"

if os.path.exists(model_path):
    with st.spinner("Loading model..."):
        with open(model_path, "rb") as file:
            model = pickle.load(file)

    st.markdown(f'''
    <div style="
        background: #FEFEFF;
        border: 1px solid #C7C7D2;
        border-radius: 24px;
        padding: 24px;
        margin-top: 8px;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    ">
        <p style="font-size:20px; color:#7F7FA4">
            Model loaded successfully!
        </p>
    </div>
    ''', unsafe_allow_html=True)

else:
    with st.spinner("Training and saving model..."):
        model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            solver=solver,
            C=C,
        )
        model.fit(X_train, y_train)
        with open(model_path, "wb") as file:
            pickle.dump(model, file)

    st.markdown(f'''
    <div style="
        background: #FEFEFF;
        border: 1px solid #C7C7D2;
        border-radius: 24px;
        padding: 24px;
        margin-top: 8px;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    ">
        <p style="font-size:20px; color:#7F7FA4">
            Model trained and saved successfully!
        </p>
    </div>
    ''', unsafe_allow_html=True)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.markdown('<div class="section"></div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle" style="margin-bottom:24px">Model Evaluation Metrics</div>', unsafe_allow_html=True)

st.markdown(f"""
<div style="
    display: inline-block;
    background: #7F7FA4;
    color: #FEFEFF;
    border-radius: 16px;
    padding: 12px 32px;
    font-size: 20px;
    font-weight: 500;
    text-align: center;
    margin-bottom: 24px;
">
    Accuracy: {accuracy*100:.1f}%
</div>
""", unsafe_allow_html=True)

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