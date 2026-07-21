import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack, csr_matrix

st.set_page_config(
    page_title="Authenticity — AI vs Human Text Detector",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

STYLE_FEATURES = ["word_count", "sentence_len_std", "lexical_diversity", "punct_density"]


@st.cache_resource
def load_models():
    nb_model = joblib.load("nb_model.pkl")
    lr_model = joblib.load("lr_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    style_scaler = joblib.load("style_scaler.pkl")
    return nb_model, lr_model, rf_model, tfidf_vectorizer, style_scaler


nb_model, lr_model, rf_model, tfidf_vectorizer, style_scaler = load_models()


# ============ FEATURE ENGINEERING (matches training notebook exactly) ============
def word_count(text):
    return len(text.split())


def sentence_length_std(text):
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    return float(np.std(lengths))


def lexical_diversity(text):
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def punctuation_density(text):
    words = text.split()
    if not words:
        return 0.0
    punct_count = len(re.findall(r"[.,;:!?\-\"'()]", text))
    return punct_count / len(words)


def classify_text(text):
    """Runs the real trained pipeline: stylometric features + TF-IDF -> all three models."""
    style_vals = pd.DataFrame([{
        "word_count": word_count(text),
        "sentence_len_std": sentence_length_std(text),
        "lexical_diversity": lexical_diversity(text),
        "punct_density": punctuation_density(text),
    }])[STYLE_FEATURES]

    tfidf_vec = tfidf_vectorizer.transform([text])
    style_scaled = style_scaler.transform(style_vals)
    combined = hstack([tfidf_vec, csr_matrix(style_scaled)])

    nb_pred = nb_model.predict(tfidf_vec)[0]
    nb_proba = nb_model.predict_proba(tfidf_vec)[0]

    lr_pred = lr_model.predict(combined)[0]
    lr_proba = lr_model.predict_proba(combined)[0]

    rf_pred = rf_model.predict(combined)[0]
    rf_proba = rf_model.predict_proba(combined)[0]

    label_map = {0: "Human", 1: "AI"}
    classes = lr_model.classes_  # same class order across models (0, 1)

    def fmt(pred, proba):
        ai_idx = list(classes).index(1)
        return {
            "label": label_map[pred],
            "ai_confidence": float(proba[ai_idx]) * 100,
        }

    return {
        "Naive Bayes": fmt(nb_pred, nb_proba),
        "Logistic Regression": fmt(lr_pred, lr_proba),
        "Random Forest": fmt(rf_pred, rf_proba),
        "style_features": style_vals.iloc[0].to_dict(),
    }


# ============ GLOBAL THEME STYLING (matches design mockup) ============
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Special+Elite&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
    --ink: #0B0F1A;
    --surface: #12182B;
    --surface-alt: #171F35;
    --border: rgba(237, 239, 245, 0.09);
    --border-strong: rgba(237, 239, 245, 0.16);
    --text-primary: #EDEFF5;
    --text-muted: #8890A6;
    --human: #E8B75F;
    --human-dim: rgba(232, 183, 95, 0.12);
    --ai: #4FD1C5;
    --ai-dim: rgba(79, 209, 197, 0.12);
}

.stApp { background-color: var(--ink); color: var(--text-primary); font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #FFFFFF !important; }
code { font-family: 'JetBrains Mono', monospace; color: var(--ai); }

.eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; letter-spacing: 0.12em;
    color: var(--ai); text-transform: uppercase; display: inline-flex; align-items: center; gap: 8px;
    border: 1px solid rgba(79,209,197,0.3); background: var(--ai-dim); padding: 6px 14px; border-radius: 20px;
    margin-bottom: 1.2rem;
}

.hero-title { font-size: 2.6rem; font-weight: 600; line-height: 1.1; margin-bottom: 0.8rem; }
.hero-title .h { color: var(--human); }
.hero-title .a { color: var(--ai); }
.hero-sub { color: var(--text-muted); font-size: 1.02rem; max-width: 620px; }

.card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 1.4rem 1.6rem; margin-bottom: 1rem;
}

.result-card {
    background: linear-gradient(135deg, rgba(232,183,95,0.10) 0%, rgba(79,209,197,0.08) 100%);
    border: 1px solid var(--border-strong); border-radius: 14px; padding: 1.4rem 1.6rem;
}
.result-label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }
.result-value { font-size: 1.7rem; font-weight: 700; color: #FFFFFF; margin: 0.3rem 0; }

.badge {
    display: inline-block; font-size: 0.72rem; font-weight: 600; padding: 0.22rem 0.65rem;
    border-radius: 20px; margin-right: 0.35rem; font-family: 'JetBrains Mono', monospace;
}
.badge-ai { background: rgba(79,209,197,0.15); color: #5EEAD4; border: 1px solid rgba(79,209,197,0.3); }
.badge-human { background: rgba(232,183,95,0.15); color: #FCD34D; border: 1px solid rgba(232,183,95,0.3); }

.model-table { width: 100%; border-collapse: collapse; margin-top: 0.6rem; }
.model-table th {
    text-align: left; padding: 10px 14px; font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.06em; text-transform: uppercase; color: var(--text-muted); border-bottom: 1px solid var(--border);
}
.model-table td { padding: 12px 14px; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
.model-table tr.winner { background: var(--ai-dim); }
.model-table .name { font-family: 'Space Grotesk', sans-serif; font-weight: 600; }

.stButton > button {
    background: var(--text-primary); color: var(--ink); font-weight: 700; border: none; border-radius: 8px;
    padding: 0.6rem 1.6rem;
}
.stButton > button:hover { background: var(--ai); color: var(--ink); }

.stTextArea textarea {
    background: var(--surface-alt) !important; color: var(--text-primary) !important;
    border: 1px solid var(--border-strong) !important; font-family: 'JetBrains Mono', monospace !important;
}

[data-testid="stMetricValue"] { color: #FFFFFF; font-family: 'Space Grotesk', sans-serif; }
[data-testid="stMetricLabel"] { color: var(--text-muted); }

.stTabs [data-baseweb="tab-list"] { gap: 4px; background: var(--surface); border-radius: 10px; padding: 4px; border: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { color: var(--text-muted); font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; border-radius: 7px; }
.stTabs [aria-selected="true"] { background: var(--ai) !important; color: #04342C !important; }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ============ HEADER ============
st.markdown('<div class="eyebrow">● 487,235 essays classified</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-title">Tell <span class="h">human</span> writing from '
    '<span class="a">AI</span> output.</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="hero-sub">A classic machine learning pipeline that scores text on writing '
    'style, not just word choice — trained on the AI vs Human Text dataset, compared across '
    'three algorithm families.</div>',
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

tab_try, tab_perf, tab_method = st.tabs(["🔍  Try it", "📊  Model performance", "🧠  Model & method"])

# ============ TAB 1: TRY IT ============
with tab_try:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    text_input = st.text_area(
        "Paste text to classify",
        height=200,
        placeholder="Paste an essay, article, or paragraph here..."
    )
    classify_clicked = st.button("🔎  Classify", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if classify_clicked:
        if not text_input or not text_input.strip():
            st.warning("Paste some text first.")
        else:
            with st.spinner("Running all three models..."):
                result = classify_text(text_input)

            lr_result = result["Logistic Regression"]
            badge_class = "badge-ai" if lr_result["label"] == "AI" else "badge-human"

            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Prediction (Logistic Regression)</div>
                    <div class="result-value">{lr_result['label']}</div>
                    <span class="badge {badge_class}">AI confidence: {lr_result['ai_confidence']:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            with r2:
                agree = sum(1 for m in ["Naive Bayes", "Logistic Regression", "Random Forest"]
                            if result[m]["label"] == lr_result["label"])
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">Model agreement</div>
                    <div class="result-value">{agree} / 3 models agree</div>
                    <span class="badge badge-ai">Logistic Regression is the primary model (99.36% F1)</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Breakdown by model")
            rows = ""
            for name in ["Logistic Regression", "Random Forest", "Naive Bayes"]:
                r = result[name]
                winner_class = "winner" if name == "Logistic Regression" else ""
                badge = "badge-ai" if r["label"] == "AI" else "badge-human"
                rows += f"""
                <tr class="{winner_class}">
                    <td class="name">{name}</td>
                    <td><span class="badge {badge}">{r['label']}</span></td>
                    <td>{r['ai_confidence']:.1f}%</td>
                </tr>
                """
            st.markdown(f"""
            <div class="card">
            <table class="model-table">
                <thead><tr><th>Model</th><th>Prediction</th><th>AI confidence</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Stylometric features (this text)")
            sf = result["style_features"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Word count", f"{sf['word_count']:.0f}")
            c2.metric("Sentence variance", f"{sf['sentence_len_std']:.2f}")
            c3.metric("Lexical diversity", f"{sf['lexical_diversity']:.2f}")
            c4.metric("Punctuation density", f"{sf['punct_density']:.2f}")

            st.caption(
                "Prediction uses the real trained pipeline (TF-IDF + 4 stylometric features → "
                "Logistic Regression / Random Forest; TF-IDF only → Naive Bayes). Not a hiring or "
                "publishing decision — a model estimate."
            )

# ============ TAB 2: MODEL PERFORMANCE ============
with tab_perf:
    m1, m2, m3 = st.columns(3)
    m1.metric("Best model F1", "99.36%")
    m2.metric("Essays after cleaning", "487,235")
    m3.metric("Errors on test set", "625 / 97,447")

    st.markdown("### Held-out test set performance")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown('<div class="card"><b>Logistic Regression</b><br>Accuracy 99.36% · F1 99.36%</div>', unsafe_allow_html=True)
    with p2:
        st.markdown('<div class="card"><b>Random Forest</b><br>Accuracy 99.15% · F1 99.15%</div>', unsafe_allow_html=True)
    with p3:
        st.markdown('<div class="card"><b>Naive Bayes</b><br>Accuracy 95.47% · F1 95.45%</div>', unsafe_allow_html=True)

    st.markdown("### Confusion matrix")
    st.image("chart_confusion_matrix.png", use_container_width=True)

    st.markdown("### Stylometric feature importance")
    st.image("chart_feature_importance.png", use_container_width=True)
    st.caption(
        "sentence_len_std ranks highest — confirms the EDA finding that AI text keeps a "
        "tighter, more uniform sentence rhythm than human writing."
    )

# ============ TAB 3: MODEL & METHOD ============
with tab_method:
    st.markdown("### Three genuinely different algorithm families")
    st.markdown("""
    <table class="model-table">
        <thead><tr><th>Model</th><th>Family</th><th>Features</th><th>Accuracy</th><th>F1</th></tr></thead>
        <tbody>
            <tr class="winner">
                <td class="name">Logistic Regression <span class="badge badge-ai">winner</span></td>
                <td>Linear</td><td>TF-IDF + stylometric</td><td>99.36%</td><td>99.36%</td>
            </tr>
            <tr>
                <td class="name">Random Forest</td>
                <td>Tree ensemble</td><td>TF-IDF + stylometric</td><td>99.15%</td><td>99.15%</td>
            </tr>
            <tr>
                <td class="name">Naive Bayes</td>
                <td>Probabilistic</td><td>TF-IDF only</td><td>95.47%</td><td>95.45%</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("### Pipeline")
    steps = [
        ("1. Clean", "Drop duplicate and null rows. 487,235 essays remain from the raw file."),
        ("2. EDA", "Check class balance, length, and sentence structure before building anything."),
        ("3. Engineer", "Four stylometric features kept from EDA. TF-IDF (5,000 features) built alongside."),
        ("4. Combine", "Stylometric features scaled and stacked onto the TF-IDF matrix for two of the three models."),
        ("5. Compare", "Same train/test split across all three, scored on accuracy, precision, recall, F1."),
    ]
    cols = st.columns(5)
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(f'<div class="card"><b>{title}</b><br><span style="color:#8890A6;font-size:0.85rem;">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown("### Key engineering decisions")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
        <div class="card">
        <b>Why these four stylometric features</b><br><br>
        EDA tested six candidates. Average sentence length barely differed between classes and
        was dropped. Sentence length variance, word count, lexical diversity, and punctuation
        density all showed real separation and made the final feature set.
        </div>
        <div class="card">
        <b>Why Naive Bayes only gets TF-IDF</b><br><br>
        Multinomial Naive Bayes assumes non-negative, count-like input. The scaled, centered
        stylometric features (some negative after standardization) don't fit that assumption,
        so Naive Bayes trains on word content alone.
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="card">
        <b>Why class_weight is balanced</b><br><br>
        The dataset is 62.8% human, 37.2% AI — not severe, but enough to bias a model toward
        the majority class without correction.
        </div>
        <div class="card">
        <b>Why TF-IDF is capped at 5,000 features</b><br><br>
        A larger vocabulary would help Naive Bayes, but Random Forest gets slow and
        memory-heavy on very high-dimensional sparse input. 5,000 keeps every model trainable
        on the same feature set within a normal Colab session.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Honest limitations")
    l1, l2, l3 = st.columns(3)
    limits = [
        ("Shortcut risk", "word_count is the second-strongest stylometric feature. Part of what the model learned is genuinely about writing style — part of it is this dataset's AI outputs tending to run shorter."),
        ("Single dataset", "Trained and tested on one AI vs Human Text dataset. A different AI model family or writing domain could shift these numbers."),
        ("Simple sentence splitting", "Sentence boundaries use a basic regex on . ! ? — not a proper NLP tokenizer."),
    ]
    for col, (title, desc) in zip([l1, l2, l3], limits):
        with col:
            st.markdown(f'<div class="card"><span class="badge badge-human">{title}</span><br><br><span style="color:#8890A6;font-size:0.85rem;">{desc}</span></div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("scikit-learn · polars · TF-IDF · streamlit · trained on Google Colab")
