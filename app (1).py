import streamlit as st
import joblib

st.set_page_config(
    page_title="Complaint Product Classifier",
    page_icon="📋",
    layout="wide"
)

@st.cache_resource
def load_model():
    clf = joblib.load("complaint_product_classifier.pkl")
    vectorizer = joblib.load("hashing_vectorizer.pkl")
    return clf, vectorizer

clf, vectorizer = load_model()

# ---- Header ----
col_icon, col_title = st.columns([0.06, 0.94])
with col_icon:
    st.markdown("### 📋")
with col_title:
    st.markdown("## Consumer Complaint Product Classifier")

st.caption(
    "Student project · CFPB Consumer Complaint Database (3.8M+ complaints, "
    "9 consolidated product categories). **Not an official CFPB tool — "
    "for demo/research purposes only.**"
)

st.divider()

tab1, tab2, tab3 = st.tabs(["🔍 Classify a Complaint", "📊 Model Performance", "🧠 Model & Method"])

# ---- TAB 1: Classify ----
with tab1:
    st.markdown("### Classify a complaint")
    st.write(
        "Paste a consumer complaint narrative below. The model predicts which "
        "financial product category it most likely belongs to, based on the wording alone."
    )

    text_input = st.text_area(
        "Complaint narrative",
        height=180,
        placeholder="e.g. I have been trying to get my credit report corrected for months and no one responds..."
    )

    predict_clicked = st.button("Predict Category", type="primary")

    if predict_clicked:
        if text_input.strip() == "":
            st.warning("Please enter a complaint narrative first.")
        else:
            X = vectorizer.transform([text_input])
            prediction = clf.predict(X)[0]

            st.markdown("#### Result")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Predicted category", prediction)
            with m2:
                st.metric("Model", "SGDClassifier (linear SVM)")

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
