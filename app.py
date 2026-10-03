import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import confusion_matrix


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TruthLens",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f8fafc;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4, h5, h6 {
    color: #111827 !important;
}

p {
    color: #374151;
}

textarea {
    background-color: #ffffff !important;
    color: #111827 !important;
    border-radius: 10px !important;
}

div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

[data-testid="stMetricLabel"] {
    color: #4b5563 !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #111827 !important;
    font-weight: 800;
}

.info-box {
    background-color: #ffffff;
    border-left: 5px solid #2563eb;
    padding: 18px;
    border-radius: 10px;
    margin: 10px 0;
    color: #111827;
}

.warning-box {
    background-color: #fff7ed;
    border-left: 5px solid #f97316;
    padding: 18px;
    border-radius: 10px;
    margin: 10px 0;
    color: #111827;
}

.success-box {
    background-color: #f0fdf4;
    border-left: 5px solid #22c55e;
    padding: 18px;
    border-radius: 10px;
    margin: 10px 0;
    color: #111827;
}

.danger-box {
    background-color: #fef2f2;
    border-left: 5px solid #ef4444;
    padding: 18px;
    border-radius: 10px;
    margin: 10px 0;
    color: #111827;
}

.footer {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 20px;
    margin-top: 30px;
    text-align: center;
    color: #4b5563;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL FILES
# ============================================================

MODEL_FILE = "fake_news_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"
METRICS_FILE = "model_metrics.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_resource
def load_vectorizer():
    return joblib.load(VECTORIZER_FILE)


@st.cache_data
def load_metrics():
    return joblib.load(METRICS_FILE)


model = load_model()
vectorizer = load_vectorizer()
metrics = load_metrics()


# ============================================================
# SAFE METRIC HELPER
# ============================================================

def get_metric(metrics_dict, *possible_names):
    for name in possible_names:
        if name in metrics_dict:
            return metrics_dict[name]
    return 0


accuracy = get_metric(metrics, "accuracy")
precision = get_metric(metrics, "precision")
recall = get_metric(metrics, "recall")
f1 = get_metric(
    metrics,
    "f1",
    "f1_score",
    "f1score"
)


# ============================================================
# DATASET STATISTICS
# ============================================================

total_articles = 44898
fake_articles = 23481
real_articles = 21417

training_size = 35918
testing_size = 8980


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion_matrix_values = [
    [4619, 77],
    [46, 4238]
]


# ============================================================
# XAI FUNCTION
# ============================================================

def get_explanation(
    text_vector,
    model,
    vectorizer,
    prediction,
    top_n=8
):

    feature_names = vectorizer.get_feature_names_out()

    coefficients = model.coef_[0]

    values = text_vector.toarray()[0]

    contributions = values * coefficients

    feature_data = pd.DataFrame({
        "Word": feature_names,
        "Contribution": contributions
    })

    feature_data = feature_data[
        feature_data["Contribution"] != 0
    ]

    if prediction == "REAL":

        supporting = feature_data[
            feature_data["Contribution"] > 0
        ].sort_values(
            "Contribution",
            ascending=False
        ).head(top_n)

        opposing = feature_data[
            feature_data["Contribution"] < 0
        ].sort_values(
            "Contribution",
            ascending=True
        ).head(top_n)

    else:

        supporting = feature_data[
            feature_data["Contribution"] < 0
        ].sort_values(
            "Contribution",
            ascending=True
        ).head(top_n)

        opposing = feature_data[
            feature_data["Contribution"] > 0
        ].sort_values(
            "Contribution",
            ascending=False
        ).head(top_n)

    return supporting, opposing


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("""
<div style="
    padding: 10px 0 25px 0;
    text-align: center;
">
    <div style="
        font-size:28px;
        font-weight:800;
        color:#ffffff;
        letter-spacing:0.5px;
    ">
        TruthLens
    </div>

    <div style="
        font-size:11px;
        color:#9ca3af;
        margin-top:6px;
        letter-spacing:1.2px;
        font-weight:600;
    ">
        AI-POWERED NEWS ANALYSIS
    </div>
</div>
""", unsafe_allow_html=True)


st.sidebar.markdown("---")


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔍 News Analyzer",
        "📈 Dataset Analytics",
        "📊 Model Information",
        "ℹ️ About Project"
    ]
)


st.sidebar.markdown("---")


st.sidebar.markdown(
    """
    <div style="
        text-align:center;
        color:#9ca3af;
        font-size:12px;
        line-height:1.6;
    ">
        <b>TruthLens</b><br>
        AI-Powered Fake News Detection<br>
        & Explainable Analysis
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🔎 TruthLens")

    st.subheader(
        "AI-Powered Fake News Detection & Explainable Analysis"
    )

    st.markdown(
        """
        TruthLens uses machine learning and Natural Language Processing
        to analyze news articles and estimate whether they are likely to
        be **FAKE** or **REAL**.
        """
    )

    st.markdown("---")

    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Articles",
            f"{total_articles:,}"
        )

    with col2:
        st.metric(
            "Fake Articles",
            f"{fake_articles:,}"
        )

    with col3:
        st.metric(
            "Real Articles",
            f"{real_articles:,}"
        )

    with col4:
        st.metric(
            "Training Articles",
            f"{training_size:,}"
        )

    st.markdown("---")

    st.subheader("🤖 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )

    st.markdown("---")

    st.subheader("⚡ How TruthLens Works")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            ### 1️⃣ Input

            Enter or paste a news article into the
            **News Analyzer**.
            """
        )

    with col2:

        st.markdown(
            """
            ### 2️⃣ AI Analysis

            TF-IDF converts the article into numerical
            features and the trained Logistic Regression
            model analyzes the text.
            """
        )

    with col3:

        st.markdown(
            """
            ### 3️⃣ Explanation

            TruthLens displays the prediction,
            confidence, probability and important words
            influencing the result.
            """
        )

    st.markdown("---")

    st.markdown(
        """
        <div class="warning-box">
        <b>⚠️ Important:</b><br>
        TruthLens is a machine-learning demonstration project.
        A prediction of REAL does not prove that an article is
        factually true, and a prediction of FAKE does not by itself
        establish that an article is false.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# NEWS ANALYZER
# ============================================================

elif page == "🔍 News Analyzer":

    st.title("🔍 News Analyzer")

    st.write(
        "Paste a news article below to analyze it using the trained AI model."
    )

    article_text = st.text_area(
        "Enter News Article",
        height=280,
        placeholder=(
            "Paste the full news article here..."
        )
    )

    analyze_button = st.button(
        "🔎 Analyze Article",
        use_container_width=True
    )

    if analyze_button:

        if not article_text.strip():

            st.warning(
                "Please enter a news article before analyzing."
            )

        else:

            # ------------------------------------------------
            # TEXT VECTOR
            # ------------------------------------------------

            text_vector = vectorizer.transform(
                [article_text]
            )

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            prediction = model.predict(
                text_vector
            )[0]

            probabilities = model.predict_proba(
                text_vector
            )[0]

            classes = model.classes_

            probability_dict = {
                classes[i]: probabilities[i]
                for i in range(len(classes))
            }

            fake_probability = probability_dict.get(
                "FAKE",
                0
            )

            real_probability = probability_dict.get(
                "REAL",
                0
            )

            confidence = max(
                fake_probability,
                real_probability
            )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            st.markdown("---")

            st.subheader("🎯 Prediction Result")

            if prediction == "FAKE":

                st.markdown(
                    f"""
                    <div class="danger-box">
                    <h2>🚨 FAKE NEWS</h2>
                    <p>
                    The model classified this article as
                    <b>FAKE</b>.
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="success-box">
                    <h2>✅ REAL NEWS</h2>
                    <p>
                    The model classified this article as
                    <b>REAL</b>.
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            st.subheader("📊 Prediction Confidence")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Prediction",
                    prediction
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

            with col3:

                st.metric(
                    "Article Words",
                    f"{len(article_text.split()):,}"
                )

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            st.markdown("---")

            st.subheader("📈 Prediction Probabilities")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "FAKE Probability",
                    f"{fake_probability * 100:.2f}%"
                )

                st.progress(
                    float(fake_probability)
                )

            with col2:

                st.metric(
                    "REAL Probability",
                    f"{real_probability * 100:.2f}%"
                )

                st.progress(
                    float(real_probability)
                )

            # ------------------------------------------------
            # ARTICLE STATISTICS
            # ------------------------------------------------

            st.markdown("---")

            st.subheader("📝 Article Statistics")

            words = article_text.split()

            sentences = [
                s for s in article_text.split(".")
                if s.strip()
            ]

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Words",
                    f"{len(words):,}"
                )

            with col2:

                st.metric(
                    "Characters",
                    f"{len(article_text):,}"
                )

            with col3:

                st.metric(
                    "Sentences",
                    f"{len(sentences):,}"
                )

            # ------------------------------------------------
            # EXPLAINABLE AI
            # ------------------------------------------------

            st.markdown("---")

            st.subheader(
                "🧠 Explainable AI"
            )

            st.write(
                "The following words contributed most strongly "
                "to the model's prediction."
            )

            supporting, opposing = get_explanation(
                text_vector,
                model,
                vectorizer,
                prediction
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "### 🔵 Supporting Words"
                )

                if len(supporting) > 0:

                    display_supporting = supporting.copy()

                    display_supporting[
                        "Contribution"
                    ] = display_supporting[
                        "Contribution"
                    ].round(4)

                    st.dataframe(
                        display_supporting,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "No strong supporting words were found."
                    )

            with col2:

                st.markdown(
                    "### 🟠 Opposing Words"
                )

                if len(opposing) > 0:

                    display_opposing = opposing.copy()

                    display_opposing[
                        "Contribution"
                    ] = display_opposing[
                        "Contribution"
                    ].round(4)

                    st.dataframe(
                        display_opposing,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "No strong opposing words were found."
                    )

            st.markdown("---")

            st.markdown(
                """
                <div class="warning-box">
                <b>⚠️ Interpretation:</b><br>
                The highlighted words show which terms influenced
                the machine-learning model. They are not independent
                evidence that the article is true or false.
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# DATASET ANALYTICS
# ============================================================

elif page == "📈 Dataset Analytics":

    st.title("📈 Dataset Analytics")

    st.write(
        "Overview of the dataset used to train and evaluate the model."
    )

    st.markdown("---")

    st.subheader("📊 Dataset Distribution")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Articles",
            f"{total_articles:,}"
        )

    with col2:

        st.metric(
            "FAKE Articles",
            f"{fake_articles:,}"
        )

    with col3:

        st.metric(
            "REAL Articles",
            f"{real_articles:,}"
        )

    st.markdown("---")

    # Dataset distribution table

    distribution_df = pd.DataFrame({
        "Category": [
            "FAKE",
            "REAL"
        ],
        "Articles": [
            fake_articles,
            real_articles
        ]
    })

    st.dataframe(
        distribution_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("🧪 Training vs Testing Data")

    train_test_df = pd.DataFrame({
        "Dataset": [
            "Training",
            "Testing"
        ],
        "Articles": [
            training_size,
            testing_size
        ],
        "Percentage": [
            f"{training_size / total_articles * 100:.2f}%",
            f"{testing_size / total_articles * 100:.2f}%"
        ]
    })

    st.dataframe(
        train_test_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("📌 Dataset Information")

    st.markdown(
        """
        <div class="info-box">

        <b>Dataset size:</b> 44,898 articles<br><br>

        <b>Fake articles:</b> 23,481<br><br>

        <b>Real articles:</b> 21,417<br><br>

        <b>Training data:</b> 35,918 articles<br><br>

        <b>Testing data:</b> 8,980 articles<br><br>

        <b>Train/Test split:</b> 80% / 20%

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "📊 Model Information":

    st.title("📊 Model Information")

    st.write(
        "Technical information about the machine-learning model."
    )

    st.markdown("---")

    st.subheader("🤖 Machine Learning Model")

    model_info = pd.DataFrame({
        "Component": [
            "Algorithm",
            "Feature Extraction",
            "Vectorizer",
            "Maximum Features",
            "Training Articles",
            "Testing Articles"
        ],
        "Value": [
            "Logistic Regression",
            "TF-IDF",
            "TfidfVectorizer",
            "50,000",
            f"{training_size:,}",
            f"{testing_size:,}"
        ]
    })

    st.dataframe(
        model_info,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("📈 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{f1 * 100:.2f}%"
        )

    st.markdown("---")

    st.subheader("🔢 Confusion Matrix")

    cm_df = pd.DataFrame(
        confusion_matrix_values,
        index=[
            "Actual FAKE",
            "Actual REAL"
        ],
        columns=[
            "Predicted FAKE",
            "Predicted REAL"
        ]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("📚 Metric Explanation")

    st.markdown(
        """
        **Accuracy**  
        Measures the proportion of all predictions that were correct.

        **Precision**  
        Measures how many articles predicted as a particular class
        actually belonged to that class.

        **Recall**  
        Measures how many articles belonging to a class were
        successfully identified.

        **F1 Score**  
        Combines precision and recall into a single metric.
        """
    )

    st.markdown("---")

    st.subheader("🧪 Training vs Testing Data")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Training Articles",
            f"{training_size:,}"
        )

    with col2:

        st.metric(
            "Testing Articles",
            f"{testing_size:,}"
        )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About TruthLens")

    st.subheader(
        "AI-Powered Fake News Detection & Explainable Analysis"
    )

    st.markdown("---")

    st.markdown(
        """
        ### 🎯 Project Objective

        TruthLens is a machine-learning based application designed
        to analyze news text and classify it as either **FAKE** or
        **REAL** based on patterns learned from a labeled dataset.
        """
    )

    st.markdown(
        """
        ### 🧠 Technologies Used

        - Python
        - Streamlit
        - Pandas
        - Scikit-learn
        - TF-IDF
        - Logistic Regression
        - Joblib
        """
    )

    st.markdown(
        """
        ### 🔬 Explainable AI

        TruthLens does not only provide a prediction.

        It also identifies words that contributed to the model's
        classification. This provides a basic form of explainable
        artificial intelligence and helps users understand how the
        model reached its prediction.
        """
    )

    st.markdown(
        """
        ### 📊 Model

        The application uses:

        **TF-IDF → Logistic Regression**

        TF-IDF converts text into numerical features, while Logistic
        Regression uses those features to classify the article.
        """
    )

    st.markdown(
        """
        ### ⚠️ Disclaimer

        TruthLens is an educational machine-learning project.

        Its predictions should not be treated as definitive proof
        that a news article is true or false. Users should verify
        important information using reliable sources and professional
        fact-checking resources.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="footer">

        <b>TruthLens</b><br>

        AI-Powered News Analysis<br><br>

        Built with Python, Streamlit & Machine Learning

        </div>
        """,
        unsafe_allow_html=True
    )
