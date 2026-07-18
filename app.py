import streamlit as st
import joblib
import numpy as np

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
    text_input = st.text_area(
        "Complaint narrative",
        height=160,
        placeholder="e.g. My car was repossessed even though I had an approved payment plan with the lender...",
        label_visibility="collapsed"
    )

    predict_clicked = st.button("🔎 Classify My Complaint", type="primary", use_container_width=False)

    if predict_clicked:
        if text_input.strip() == "":
            st.warning("Type your complaint first — then hit classify.")
        else:
            X = vectorizer.transform([text_input])
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
                "Overall model accuracy across all categories is 85.3% (see Model Performance tab)."
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
    st.markdown("### How this was built")

    st.markdown("""
**Data:** ~3.8M CFPB consumer complaints with a non-empty narrative, out of an original
8GB / 3.8M+ row dataset. The original 21 `Product` categories were consolidated into 9,
merging labels that were renamed over time by CFPB's evolving taxonomy (e.g. three
different historical labels for "Credit reporting" were merged into one).

**Vectorization:** `HashingVectorizer` (524,288 features, unigrams + bigrams) — chosen
over TF-IDF because it doesn't store a vocabulary in memory, keeping memory usage
constant regardless of dataset size.

**Model:** `SGDClassifier` (hinge loss, equivalent to a linear SVM), trained
out-of-core via `partial_fit` in batches of 30,000 rows over 3 epochs, with manually
computed balanced class weights to account for the ~55% share held by the largest
category (Credit reporting).

**Confidence scores:** since hinge loss doesn't produce calibrated probabilities,
per-prediction confidence is computed by applying softmax to the raw decision scores
across all 9 categories — a reasonable proxy for relative confidence, not a true
statistical probability.
    """)

    st.markdown("#### Honest limitations")
    st.markdown("""
- **Bag-of-words ceiling:** additional training epochs and bigrams did not meaningfully
  improve accuracy (85.26% → 85.31%), suggesting remaining errors are semantic in nature
  rather than a training issue — a linear model can't distinguish "vehicle loan default"
  from "personal loan default" language without deeper contextual understanding.
- **Minority class performance:** Payday/title/personal loan (F1 0.44) and Vehicle loan
  (F1 0.55) are the weakest categories, both due to smaller sample sizes and genuine
  vocabulary overlap with Debt collection and each other.
- **Text-only input:** the model only sees the narrative — it doesn't use other fields
  (state, company, submission channel) that a production system might also leverage.
    """)

    st.caption("Model: scikit-learn SGDClassifier · random_state=42 · trained on Google Colab")
