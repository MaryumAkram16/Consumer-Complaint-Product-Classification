import streamlit as st
import joblib
import numpy as np
import speech_recognition as sr
from deep_translator import GoogleTranslator

st.set_page_config(
    page_title="Complaint Classifier",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    clf = joblib.load("complaint_product_classifier.pkl")
    vectorizer = joblib.load("hashing_vectorizer.pkl")
    return clf, vectorizer

clf, vectorizer = load_model()

# ============ GLOBAL DARK THEME STYLING ============
st.markdown("""
<style>
:root {
    --bg-main: #0A0E1A;
    --bg-card: #131829;
    --bg-card-alt: #1A2036;
    --border: #262D45;
    --purple: #8B5CF6;
    --purple-dark: #6D28D9;
    --teal: #2DD4BF;
    --text-main: #E8EAF0;
    --text-muted: #8891A8;
}

.stApp {
    background-color: var(--bg-main);
    color: var(--text-main);
}

section[data-testid="stSidebar"] {
    background-color: #0D1220;
    border-right: 1px solid var(--border);
}

h1, h2, h3, h4, h5, p, span, div, label {
    color: var(--text-main);
}

/* Eyebrow label */
.eyebrow {
    display: inline-block;
    color: var(--teal);
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    font-weight: 600;
    text-transform: uppercase;
    background: rgba(45, 212, 191, 0.1);
    border: 1px solid rgba(45, 212, 191, 0.3);
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* Hero section */
.hero {
    padding: 1rem 0 2rem 0;
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 1rem;
    color: #FFFFFF;
}
.hero p {
    color: var(--text-muted);
    font-size: 1.05rem;
    max-width: 640px;
    line-height: 1.6;
}

/* Card */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Badge pills */
.badge {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    margin-right: 0.4rem;
}
.badge-purple { background: rgba(139, 92, 246, 0.15); color: #C4B5FD; border: 1px solid rgba(139, 92, 246, 0.3); }
.badge-teal { background: rgba(45, 212, 191, 0.15); color: #5EEAD4; border: 1px solid rgba(45, 212, 191, 0.3); }
.badge-warn { background: rgba(245, 158, 11, 0.15); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.3); }

/* Result card */
.result-card {
    background: linear-gradient(135deg, rgba(139,92,246,0.12) 0%, rgba(45,212,191,0.08) 100%);
    border: 1px solid rgba(139, 92, 246, 0.35);
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-top: 1rem;
}
.result-label { font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; }
.result-value { font-size: 1.8rem; font-weight: 800; color: #FFFFFF; margin: 0.3rem 0; }

/* Confidence bar */
.conf-row { margin-bottom: 0.9rem; }
.conf-label { display: flex; justify-content: space-between; font-size: 0.9rem; margin-bottom: 0.3rem; }
.conf-track { background: #1E2438; border-radius: 8px; height: 10px; overflow: hidden; }
.conf-fill { background: linear-gradient(90deg, var(--purple), var(--teal)); height: 100%; border-radius: 8px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--purple), var(--purple-dark));
    color: white;
    border: none;
    font-weight: 700;
    border-radius: 8px;
    padding: 0.6rem 1.5rem;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #9D6FFF, var(--purple));
    color: white;
}

/* Text areas / inputs */
.stTextArea textarea, .stTextInput input {
    background: var(--bg-card-alt) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Metric override */
[data-testid="stMetricValue"] { color: #FFFFFF; }
[data-testid="stMetricLabel"] { color: var(--text-muted); }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ============ SIDEBAR NAVIGATION ============
with st.sidebar:
    st.markdown("## 📋 Complaint AI")
    st.caption("Consumer complaint intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔍  Try It", "📊  Model Performance", "🧠  Model & Method"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Student project · CFPB dataset\nNot an official CFPB tool")

# ============ TRY IT ============
if page == "🔍  Try It":
    st.markdown('<div class="eyebrow">● TRAINED ON 3.8M REAL COMPLAINTS</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero">
        <h1>Got a complaint?<br>Let's find out where it belongs.</h1>
        <p>Type or speak your issue — in English or Urdu — and this model predicts
        which financial product category it falls under, trained on real CFPB
        consumer complaint data.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    col_lang, col_mode = st.columns(2)
    with col_lang:
        language = st.radio("Language", ["English", "Urdu (اردو)"], horizontal=True)
    with col_mode:
        input_mode = st.radio("Input method", ["⌨️ Type", "🎤 Speak"], horizontal=True)

    if language.startswith("Urdu"):
        st.markdown(
            '<span class="badge badge-warn">⚠ Translated before classification — see Model & Method for details</span>',
            unsafe_allow_html=True
        )

    raw_text = ""

    if input_mode == "⌨️ Type":
        placeholder = (
            "مثال کے طور پر: میری گاڑی قرض کی ادائیگی کے باوجود ضبط کر لی گئی..."
            if language.startswith("Urdu")
            else "e.g. My car was repossessed even though I had an approved payment plan..."
        )
        raw_text = st.text_area(
            "Complaint narrative", height=150, placeholder=placeholder,
            label_visibility="collapsed"
        )
    else:
        st.write("Tap to record, then stop when you're done speaking.")
        audio_value = st.audio_input("Record your complaint")

        if audio_value is not None:
            recognizer = sr.Recognizer()
            try:
                with sr.AudioFile(audio_value) as source:
                    audio_data = recognizer.record(source)
                lang_code = "ur-PK" if language.startswith("Urdu") else "en-US"
                raw_text = recognizer.recognize_google(audio_data, language=lang_code)
                st.text_area("Transcribed text (edit if needed)", value=raw_text, height=100, key="transcribed")
                raw_text = st.session_state.get("transcribed", raw_text)
            except sr.UnknownValueError:
                st.error("Couldn't understand the audio — try speaking again, closer to the mic.")
            except sr.RequestError:
                st.error("Speech recognition service unavailable right now. Try typing instead.")

    predict_clicked = st.button("🔎  Classify My Complaint", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        if raw_text.strip() == "":
            st.warning("Enter or record your complaint first — then hit classify.")
        else:
            if language.startswith("Urdu"):
                try:
                    english_text = GoogleTranslator(source="ur", target="en").translate(raw_text)
                    st.markdown(f'<div class="card"><b>Translated to English:</b><br>{english_text}</div>', unsafe_allow_html=True)
                except Exception:
                    st.error("Translation failed. Please try again or type in English.")
                    st.stop()
            else:
                english_text = raw_text

            X = vectorizer.transform([english_text])
            prediction = clf.predict(X)[0]

            scores = clf.decision_function(X)[0]
            exp_scores = np.exp(scores - np.max(scores))
            confidence = exp_scores / exp_scores.sum()

            classes = clf.classes_
            order = np.argsort(confidence)[::-1]

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Predicted Category</div>
                <div class="result-value">{prediction}</div>
                <span class="badge badge-purple">Confidence: {confidence[order[0]]*100:.1f}%</span>
                <span class="badge badge-teal">Linear SVM</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Confidence breakdown")
            bars_html = '<div class="card">'
            for i in order[:5]:
                pct = confidence[i] * 100
                bars_html += f"""
                <div class="conf-row">
                    <div class="conf-label"><span>{classes[i]}</span><span>{pct:.1f}%</span></div>
                    <div class="conf-track"><div class="conf-fill" style="width:{pct}%;"></div></div>
                </div>
                """
            bars_html += '</div>'
            st.markdown(bars_html, unsafe_allow_html=True)

            st.caption(
                "Confidence reflects the model's relative certainty, not a guarantee of correctness. "
                "Overall accuracy across all categories is 85.3% on English narratives."
            )

# ============ MODEL PERFORMANCE ============
elif page == "📊  Model Performance":
    st.markdown('<div class="eyebrow">● EVALUATED ON 763,939 HELD-OUT COMPLAINTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>Test set performance</h1></div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Accuracy", "85.3%")
    with m2:
        st.metric("Weighted F1", "0.86")
    with m3:
        st.metric("Macro F1", "0.72")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Confusion matrix")
    st.image("complaint-product-classification.PNG", use_container_width=True)
    st.caption(
        "Normalized by actual class. Most confusion is concentrated between categories "
        "with genuine subject overlap — e.g. Payday/title loans vs. Vehicle loans, since "
        "title loans are often secured against a vehicle."
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ============ MODEL & METHOD ============
else:
    st.markdown('<div class="eyebrow">● HOW IT WORKS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>Project overview</h1><p>A text classification system that routes consumer financial complaints to the correct product category — built to handle a real-world dataset too large to fit comfortably in memory on free-tier hardware.</p></div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Raw dataset", "8 GB / 3.8M rows")
    with m2:
        st.metric("Categories", "21 → 9")
    with m3:
        st.metric("Test accuracy", "85.3%")
    with m4:
        st.metric("Weighted F1", "0.86")

    st.markdown("### The pipeline")
    p1, p2, p3, p4, p5 = st.columns(5)
    steps = [
        ("1. Load", "Lazy-scanned the 8GB CSV with Polars, selecting only needed columns — full pandas load crashed on 12GB Colab RAM."),
        ("2. Clean", "Consolidated 21 overlapping `Product` labels into 9, merging categories renamed over time by CFPB's taxonomy changes."),
        ("3. Vectorize", "`HashingVectorizer` (524,288 features) — no stored vocabulary, so memory stays constant regardless of dataset size."),
        ("4. Train", "`SGDClassifier` (linear SVM) trained out-of-core via `partial_fit`, batches of 30K rows, 3 epochs, balanced class weights."),
        ("5. Evaluate", "Stratified 80/20 split, per-class precision/recall/F1, plus a normalized confusion matrix."),
    ]
    for col, (title, desc) in zip([p1, p2, p3, p4, p5], steps):
        with col:
            st.markdown(f'<div class="card"><b>{title}</b><br><span style="color:#8891A8;font-size:0.85rem;">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown("### Key engineering decisions")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
        <div class="card">
        <b>Why HashingVectorizer over TF-IDF</b><br><br>
        TF-IDF must build and store a full vocabulary before transforming — at millions
        of documents, that risks exhausting available RAM. HashingVectorizer maps tokens
        directly to fixed hash buckets with no stored vocabulary, keeping memory flat
        regardless of dataset size.
        </div>
        <div class="card">
        <b>Why out-of-core training</b><br><br>
        Materializing a vectorized matrix for all 3M+ rows at once would require an
        enormous sparse array in memory. Training in 30,000-row batches — vectorize,
        fit, discard, repeat — keeps peak memory constant.
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class="card">
        <b>Why manually computed class weights</b><br><br>
        <code>SGDClassifier.partial_fit()</code> doesn't support <code>class_weight='balanced'</code>
        directly since balancing requires the full class distribution upfront. Weights were
        precomputed once and passed explicitly.
        </div>
        <div class="card">
        <b>What didn't work (and why that's useful)</b><br><br>
        A 2nd/3rd training epoch and bigrams moved accuracy only 0.05 points (85.26% → 85.31%),
        indicating the model had converged — remaining errors are a representation ceiling,
        not undertraining.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### How confidence scores work")
    st.markdown("""
    <div class="card">
    Hinge loss (the linear SVM's loss function) doesn't produce calibrated probabilities.
    Per-prediction confidence on the <b>Try It</b> page is computed by applying softmax to
    the model's raw decision scores across all 9 categories — a reasonable proxy for
    relative confidence, not a true statistical probability.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Honest limitations")
    l1, l2, l3 = st.columns(3)
    limitations = [
        ("Bag-of-words ceiling", "A linear model can't distinguish 'vehicle loan default' from 'personal loan default' without deeper semantic context."),
        ("Minority class weakness", "Payday/title/personal loan (F1 0.44) and Vehicle loan (F1 0.55) suffer from fewer examples and genuine vocabulary overlap."),
        ("Text-only input", "The model only reads the narrative — it ignores other available signals like state, company, or submission channel."),
    ]
    for col, (title, desc) in zip([l1, l2, l3], limitations):
        with col:
            st.markdown(f'<div class="card"><span class="badge badge-warn">{title}</span><br><br><span style="color:#8891A8;font-size:0.9rem;">{desc}</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Model: scikit-learn SGDClassifier · random_state=42 · trained on Google Colab (free tier, ~12GB RAM)")
