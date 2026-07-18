import streamlit as st
import joblib
import numpy as np
import speech_recognition as sr
from deep_translator import GoogleTranslator

st.set_page_config(
    page_title="Complaint Classifier",
    page_icon="📋",
    layout="wide"
)

@st.cache_resource
def load_model():
    clf = joblib.load("complaint_product_classifier.pkl")
    vectorizer = joblib.load("hashing_vectorizer.pkl")
    return clf, vectorizer

clf, vectorizer = load_model()

# ---- Custom styling ----
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
    padding: 2.5rem 2rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 1.5rem;
}
.hero h1 {
    color: white;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #DBEAFE;
    font-size: 1.05rem;
    margin-bottom: 0;
}
.result-card {
    background: #F0F7FF;
    border: 1px solid #BFDBFE;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---- Hero section ----
st.markdown("""
<div class="hero">
    <h1>📋 Got a complaint? Let's find out where it belongs.</h1>
    <p>Paste your issue below — this model reads the wording and predicts which financial
    product category it falls under, trained on 3.8M+ real CFPB consumer complaints.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Try It", "📊 Model Performance", "🧠 Model & Method"])

CATEGORIES = None  # populated after first prediction from clf.classes_

# ---- TAB 1: Classify ----
with tab1:
    st.markdown("#### Describe your complaint")

    col_lang, col_mode = st.columns(2)
    with col_lang:
        language = st.radio("Language", ["English", "Urdu (اردو)"], horizontal=True)
    with col_mode:
        input_mode = st.radio("Input method", ["⌨️ Type", "🎤 Speak"], horizontal=True)

    if language.startswith("Urdu"):
        st.caption(
            "⚠️ The model was trained only on English complaints. Urdu input is "
            "automatically translated to English before classification — accuracy "
            "depends on translation quality, not just the model itself."
        )

    raw_text = ""

    if input_mode == "⌨️ Type":
        placeholder = (
            "مثال کے طور پر: میری گاڑی قرض کی ادائیگی کے باوجود ضبط کر لی گئی..."
            if language.startswith("Urdu")
            else "e.g. My car was repossessed even though I had an approved payment plan..."
        )
        raw_text = st.text_area(
            "Complaint narrative", height=160, placeholder=placeholder,
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

    predict_clicked = st.button("🔎 Classify My Complaint", type="primary")

    if predict_clicked:
        if raw_text.strip() == "":
            st.warning("Enter or record your complaint first — then hit classify.")
        else:
            # Translate to English if needed — the model only understands English
            if language.startswith("Urdu"):
                try:
                    english_text = GoogleTranslator(source="ur", target="en").translate(raw_text)
                    st.caption(f"**Translated to English:** {english_text}")
                except Exception:
                    st.error("Translation failed. Please try again or type in English.")
                    st.stop()
            else:
                english_text = raw_text

            X = vectorizer.transform([english_text])
            prediction = clf.predict(X)[0]

            # Confidence via decision_function -> softmax (hinge loss has no predict_proba)
            scores = clf.decision_function(X)[0]
            exp_scores = np.exp(scores - np.max(scores))
            confidence = exp_scores / exp_scores.sum()

            classes = clf.classes_
            order = np.argsort(confidence)[::-1]

            st.markdown(f"""
            <div class="result-card">
                <div style="font-size:0.9rem;color:#555;">Predicted category</div>
                <div style="font-size:1.6rem;font-weight:700;color:#1E3A8A;">{prediction}</div>
                <div style="font-size:0.9rem;color:#555;margin-top:0.3rem;">
                    Model confidence: <b>{confidence[order[0]]*100:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Confidence breakdown")
            st.caption("How the model split its confidence across all 9 categories for this complaint.")
            for i in order[:5]:  # show top 5 for readability
                st.write(f"{classes[i]}")
                st.progress(float(confidence[i]))

            st.caption(
                "Note: this reflects the model's relative confidence, not a guarantee of correctness. "
                "Overall model accuracy across all categories is 85.3% on English narratives "
                "(see Model Performance tab)."
            )

# ---- TAB 2: Performance ----
with tab2:
    st.markdown("### Test set performance")
    st.write("Evaluated on a held-out 20% split (763,939 complaints), stratified by category.")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Accuracy", "85.3%")
    with m2:
        st.metric("Weighted F1", "0.86")
    with m3:
        st.metric("Macro F1", "0.72")

    st.markdown("#### Confusion matrix")
    st.image("complaint-product-classification.PNG", use_container_width=True)

    st.caption(
        "Normalized by actual class. Most confusion is concentrated between categories "
        "with genuine subject overlap — e.g. Payday/title loans vs. Vehicle loans, since "
        "title loans are often secured against a vehicle."
    )

# ---- TAB 3: Model & Method ----
with tab3:
    st.markdown("### Project overview")
    st.write(
        "A text classification system that routes consumer financial complaints to the "
        "correct product category, built to handle a real-world dataset too large to fit "
        "comfortably in memory on free-tier hardware."
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Raw dataset", "8 GB / 3.8M rows")
    with m2:
        st.metric("Categories", "21 → 9")
    with m3:
        st.metric("Test accuracy", "85.3%")
    with m4:
        st.metric("Weighted F1", "0.86")

    st.divider()

    # ---- Pipeline as cards ----
    st.markdown("### The pipeline")

    p1, p2, p3, p4, p5 = st.columns(5)

    with p1:
        st.markdown("**1. Load**")
        st.caption(
            "Lazy-scanned the 8GB CSV with Polars, selecting only needed columns "
            "and filtering to non-empty narratives — full pandas load crashed on "
            "12GB Colab RAM."
        )
    with p2:
        st.markdown("**2. Clean**")
        st.caption(
            "Consolidated 21 overlapping `Product` labels into 9, merging categories "
            "renamed over time by CFPB's taxonomy changes (e.g. 3 historical labels "
            "for Credit reporting → 1)."
        )
    with p3:
        st.markdown("**3. Vectorize**")
        st.caption(
            "`HashingVectorizer` (524,288 features, unigrams + bigrams) — no stored "
            "vocabulary, so memory stays constant regardless of dataset size."
        )
    with p4:
        st.markdown("**4. Train**")
        st.caption(
            "`SGDClassifier` (linear SVM) trained out-of-core via `partial_fit`, "
            "batches of 30K rows, 3 epochs, manually balanced class weights."
        )
    with p5:
        st.markdown("**5. Evaluate**")
        st.caption(
            "Stratified 80/20 split, per-class precision/recall/F1, plus a "
            "normalized confusion matrix to trace exactly where errors cluster."
        )

    st.divider()

    # ---- Key engineering decisions ----
    st.markdown("### Key engineering decisions")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("##### Why HashingVectorizer over TF-IDF")
        st.write(
            "TF-IDF must build and store a full vocabulary before transforming — at "
            "millions of documents, that vocabulary alone risks exhausting available "
            "RAM. HashingVectorizer maps tokens directly to a fixed number of hash "
            "buckets with no stored vocabulary, keeping memory flat no matter how "
            "much text is processed."
        )
        st.markdown("##### Why out-of-core training (`partial_fit`)")
        st.write(
            "Materializing a vectorized matrix for all 3M+ training rows at once "
            "would require holding an enormous sparse array in memory. Training in "
            "30,000-row batches — vectorize, fit, discard, repeat — keeps peak memory "
            "usage constant regardless of total dataset size."
        )
    with d2:
        st.markdown("##### Why manually computed class weights")
        st.write(
            "`SGDClassifier.partial_fit()` doesn't support `class_weight='balanced'` "
            "directly, since balancing requires knowing the full class distribution "
            "upfront, which a single batch can't provide. Weights were precomputed "
            "once from the full training label distribution and passed explicitly."
        )
        st.markdown("##### What didn't work (and why that's useful)")
        st.write(
            "Adding a 2nd/3rd training epoch and bigrams moved accuracy only "
            "0.05 points (85.26% → 85.31%). This negative result indicates the "
            "linear model had already converged — remaining errors are a "
            "representation ceiling, not an undertraining problem."
        )

    st.divider()

    # ---- Confidence score explainer ----
    st.markdown("### How confidence scores work")
    st.write(
        "Hinge loss (the loss function behind a linear SVM) doesn't produce "
        "calibrated probabilities the way logistic regression does. Per-prediction "
        "confidence on the **Try It** tab is computed by applying softmax to the "
        "model's raw decision scores across all 9 categories — a reasonable proxy "
        "for relative confidence, not a true statistical probability."
    )

    st.divider()

    # ---- Honest limitations as cards ----
    st.markdown("### Honest limitations")

    l1, l2, l3 = st.columns(3)
    with l1:
        st.warning(
            "**Bag-of-words ceiling**\n\n"
            "A linear model can't distinguish 'vehicle loan default' from "
            "'personal loan default' language without deeper semantic context. "
            "This is a representation limit, not a tuning issue."
        )
    with l2:
        st.warning(
            "**Minority class weakness**\n\n"
            "Payday/title/personal loan (F1 0.44) and Vehicle loan (F1 0.55) "
            "are the weakest categories — fewer training examples plus genuine "
            "vocabulary overlap with Debt collection and each other."
        )
    with l3:
        st.warning(
            "**Text-only input**\n\n"
            "The model only reads the narrative. It ignores other available "
            "signals (state, company, submission channel) that a production "
            "system would likely also use."
        )

    st.divider()
    st.caption(
        "Model: scikit-learn SGDClassifier · random_state=42 · trained on Google Colab "
        "(free tier, ~12GB RAM) · dataset: CFPB Consumer Complaint Database"
    )
