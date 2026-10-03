
import streamlit as st
import pandas as pd
import joblib
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TruthLens",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ==============================
       MAIN APPLICATION
       ============================== */

    .stApp {
        background-color: #f8fafc;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        color: #111827;
    }

    /* Headings */

    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
    }

    /* Normal text */

    p {
        color: #374151;
    }

    /* ==============================
       SIDEBAR
       ============================== */

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: #f9fafb;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #f9fafb !important;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #f9fafb;
    }

    /* ==============================
       METRIC CARDS
       ============================== */

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

    /* ==============================
       CARDS
       ============================== */

    .card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 20px;
        font-weight: 700;
        color: #111827 !important;
        margin-bottom: 10px;
    }

    .card-text {
        font-size: 15px;
        color: #374151 !important;
        line-height: 1.7;
    }

    /* ==============================
       RESULT BOXES
       ============================== */

    .real-result {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-radius: 14px;
        padding: 22px;
        margin-top: 20px;
    }

    .fake-result {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 14px;
        padding: 22px;
        margin-top: 20px;
    }

    .result-title {
        font-size: 26px;
        font-weight: 800;
        color: #111827 !important;
        margin-bottom: 8px;
    }

    .result-text {
        font-size: 15px;
        color: #374151 !important;
        line-height: 1.6;
    }

    /* ==============================
       TEXT AREA
       ============================== */

    textarea {
        background-color: #ffffff !important;
        color: #111827 !important;
        border-radius: 10px !important;
    }

    textarea::placeholder {
        color: #6b7280 !important;
    }

    /* ==============================
       BUTTON
       ============================== */

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
    }

    /* ==============================
       DATAFRAME
       ============================== */

    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* ==============================
       HIDE STREAMLIT UI
       ============================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)

# ============================================================
# FILE NAMES
# ============================================================

MODEL_FILE = "fake_news_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"
METRICS_FILE = "model_metrics.pkl"

# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    metrics = joblib.load(METRICS_FILE)

except Exception as e:

    st.error("❌ Unable to load the trained model files.")

    st.markdown("""
    ### Required files

    Make sure these files are in the same folder as `app.py`:

    - `fake_news_model.pkl`
    - `tfidf_vectorizer.pkl`
    - `model_metrics.pkl`
    """)

    st.code(str(e))

    st.stop()

# ============================================================
# SAFE METRIC FUNCTION
# ============================================================

def get_metric(metrics_dict, *possible_names):

    for name in possible_names:

        if name in metrics_dict:

            return metrics_dict[name]

    return 0


accuracy = get_metric(
    metrics,
    "accuracy"
)

precision = get_metric(
    metrics,
    "precision"
)

recall = get_metric(
    metrics,
    "recall"
)

f1 = get_metric(
    metrics,
    "f1",
    "f1_score",
    "f1score"
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion_matrix = metrics.get(
    "confusion_matrix",
    [[0, 0], [0, 0]]
)

# ============================================================
# DATASET INFORMATION
# ============================================================

# These values come from your trained dataset.
# They are displayed without requiring the large CSV files.

total_articles = 44898
fake_articles = 23481
real_articles = 21417

training_size = 35918
testing_size = 8980

# ============================================================
# EXPLAINABLE AI
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

with st.sidebar:

    st.markdown("""
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

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔍 News Analyzer",
            "📈 Dataset Analytics",
            "📊 Model Information",
            "ℹ️ About Project"
        ]
    )

    st.markdown("---")

    st.markdown("""
    <div style="
        text-align:center;
        font-size:12px;
        color:#9ca3af;
        padding-top:10px;
    ">
        TruthLens v1.0
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("TruthLens")

    st.markdown(
        "### AI-Powered Fake News Detection & Explainable Analysis"
    )

    st.write(
        "Analyze news articles using machine learning and "
        "understand which words influenced the prediction."
    )

    st.markdown("---")

    # ==============================
    # PERFORMANCE
    # ==============================

    st.subheader("🎯 Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🎯 Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "🎯 Precision",
            f"{precision * 100:.2f}%"
        )

    with col3:

        st.metric(
            "🎯 Recall",
            f"{recall * 100:.2f}%"
        )

    with col4:

        st.metric(
            "🎯 F1 Score",
            f"{f1 * 100:.2f}%"
        )

    st.markdown("")

    # ==============================
    # DATASET
    # ==============================

    st.subheader("📚 Dataset Overview")

    col1, col2, col3 = st.columns(3)

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

    st.markdown("")

    # ==============================
    # HOW IT WORKS
    # ==============================

    st.markdown("""
    <div class="card">

        <div class="card-title">
            🧠 How TruthLens Works
        </div>

        <div class="card-text">

            <b>1. News Article</b><br>
            The user enters a news article into the analyzer.

            <br><br>

            <b>2. TF-IDF Processing</b><br>
            The article is converted into numerical features
            using TF-IDF vectorization.

            <br><br>

            <b>3. Machine Learning</b><br>
            A Logistic Regression model analyzes the features.

            <br><br>

            <b>4. Prediction</b><br>
            TruthLens predicts whether the article is likely
            FAKE or REAL.

            <br><br>

            <b>5. Explainable AI</b><br>
            The system shows words that contributed to the prediction.

        </div>

    </div>
    """, unsafe_allow_html=True)

    st.info(
        "⚠️ TruthLens is an educational machine-learning project. "
        "Its prediction should not be treated as a definitive fact-check."
    )


# ============================================================
# NEWS ANALYZER
# ============================================================

elif page == "🔍 News Analyzer":

    st.title("🔍 News Analyzer")

    st.write(
        "Paste a news article below and let TruthLens analyze it."
    )

    article = st.text_area(
        "📰 News Article",
        height=280,
        placeholder="Paste the complete news article here..."
    )

    analyze = st.button(
        "🔎 Analyze News",
        use_container_width=True
    )

    if analyze:

        if not article.strip():

            st.warning(
                "Please enter a news article before analyzing."
            )

        else:

            # ==============================
            # VECTORIZE
            # ==============================

            text_vector = vectorizer.transform(
                [article]
            )

            # ==============================
            # PREDICTION
            # ==============================

            prediction = model.predict(
                text_vector
            )[0]

            # ==============================
            # PROBABILITY
            # ==============================

            probabilities = model.predict_proba(
                text_vector
            )[0]

            classes = model.classes_

            probability_dict = dict(
                zip(
                    classes,
                    probabilities
                )
            )

            confidence = max(
                probabilities
            ) * 100

            # ==============================
            # RESULT
            # ==============================

            if prediction == "REAL":

                st.markdown(
                    f"""
                    <div class="real-result">

                        <div class="result-title">
                            ✅ Likely REAL News
                        </div>

                        <div class="result-text">
                            TruthLens classified this article as
                            <b>REAL</b>.

                            <br><br>

                            Model confidence:
                            <b>{confidence:.2f}%</b>
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="fake-result">

                        <div class="result-title">
                            ⚠️ Likely FAKE News
                        </div>

                        <div class="result-text">
                            TruthLens classified this article as
                            <b>FAKE</b>.

                            <br><br>

                            Model confidence:
                            <b>{confidence:.2f}%</b>
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("")

            # ==============================
            # PROBABILITIES
            # ==============================

            st.subheader("📊 Prediction Probabilities")

            col1, col2 = st.columns(2)

            with col1:

                fake_probability = probability_dict.get(
                    "FAKE",
                    0
                )

                st.metric(
                    "FAKE Probability",
                    f"{fake_probability * 100:.2f}%"
                )

                st.progress(
                    float(fake_probability)
                )

            with col2:

                real_probability = probability_dict.get(
                    "REAL",
                    0
                )

                st.metric(
                    "REAL Probability",
                    f"{real_probability * 100:.2f}%"
                )

                st.progress(
                    float(real_probability)
                )

            # ==============================
            # ARTICLE STATISTICS
            # ==============================

            st.markdown("---")

            st.subheader("📄 Article Statistics")

            words = article.split()

            characters = len(article)

            sentences = max(
                1,
                article.count(".")
                + article.count("!")
                + article.count("?")
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Words",
                    f"{len(words):,}"
                )

            with col2:

                st.metric(
                    "Characters",
                    f"{characters:,}"
                )

            with col3:

                st.metric(
                    "Sentences",
                    f"{sentences:,}"
                )

            # ==============================
            # XAI
            # ==============================

            st.markdown("---")

            st.subheader("🧠 Explainable AI")

            st.write(
                "These words had the strongest influence on "
                "the model's prediction for this article."
            )

            supporting, opposing = get_explanation(
                text_vector,
                model,
                vectorizer,
                prediction
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("### 🔵 Supporting Words")

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

                st.markdown("### 🟠 Opposing Words")

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

            st.caption(
                "ℹ️ Explanation values show how the trained model "
                "weighted words in this particular article. "
                "They do not prove that an article is factually true or false."
            )


# ============================================================
# DATASET ANALYTICS
# ============================================================

elif page == "📈 Dataset Analytics":

    st.title("📈 Dataset Analytics")

    st.write(
        "Overview of the dataset used to train the TruthLens model."
    )

    st.markdown("---")

    # ==============================
    # DATASET KPIs
    # ==============================

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

    st.markdown("")

    # ==============================
    # DISTRIBUTION
    # ==============================

    st.subheader("📰 News Distribution")

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

    st.bar_chart(
        distribution_df.set_index("Category")
    )

    st.markdown("---")

    # ==============================
    # TRAINING VS TESTING
    # ==============================

    st.subheader("🧪 Training vs Testing Data")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Training Articles",
            f"{training_size:,}"
        )

        st.progress(
            training_size / total_articles
        )

        st.caption(
            "80% of the dataset was used for training."
        )

    with col2:

        st.metric(
            "Testing Articles",
            f"{testing_size:,}"
        )

        st.progress(
            testing_size / total_articles
        )

        st.caption(
            "20% of the dataset was used for testing."
        )

    st.markdown("---")

    st.info(
        "The deployed version does not require the original CSV "
        "files to perform predictions. Dataset statistics are "
        "stored as project information."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "📊 Model Information":

    st.title("📊 Model Information")

    st.write(
        "Technical information about the TruthLens machine-learning model."
    )

    st.markdown("---")

    # ==============================
    # MODEL
    # ==============================

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="card">

            <div class="card-title">
                🤖 Machine Learning Model
            </div>

            <div class="card-text">

                <b>Algorithm:</b> Logistic Regression

                <br><br>

                <b>Feature Extraction:</b> TF-IDF

                <br><br>

                <b>Maximum Features:</b> 50,000

                <br><br>

                <b>Training Split:</b> 80%

                <br><br>

                <b>Testing Split:</b> 20%

            </div>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="card">

            <div class="card-title">
                📚 Dataset Information
            </div>

            <div class="card-text">

                <b>Total Articles:</b>
                {total_articles:,}

                <br><br>

                <b>FAKE Articles:</b>
                {fake_articles:,}

                <br><br>

                <b>REAL Articles:</b>
                {real_articles:,}

                <br><br>

                <b>Training Articles:</b>
                {training_size:,}

                <br><br>

                <b>Testing Articles:</b>
                {testing_size:,}

            </div>

        </div>
        """, unsafe_allow_html=True)

    # ==============================
    # PERFORMANCE
    # ==============================

    st.markdown("---")

    st.subheader("🎯 Model Performance")

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

    # ==============================
    # CONFUSION MATRIX
    # ==============================

    st.markdown("---")

    st.subheader("🔢 Confusion Matrix")

    try:

        cm_df = pd.DataFrame(
            confusion_matrix,
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

        st.info(
            "The confusion matrix shows how many FAKE and REAL "
            "articles were correctly or incorrectly classified."
        )

    except Exception:

        st.info(
            "Confusion matrix information is unavailable."
        )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About TruthLens")

    st.markdown(
        "### AI-Powered Fake News Detection & Explainable Analysis"
    )

    st.markdown("---")

    # ==============================
    # OBJECTIVE
    # ==============================

    st.markdown("""
    <div class="card">

        <div class="card-title">
            🎯 Project Objective
        </div>

        <div class="card-text">

            TruthLens is a machine-learning based application
            designed to analyze news articles and classify them
            as likely <b>FAKE</b> or <b>REAL</b>.

            <br><br>

            The project also provides Explainable AI features
            that help users understand which words influenced
            the model's prediction.

        </div>

    </div>
    """, unsafe_allow_html=True)

    # ==============================
    # TECHNOLOGIES
    # ==============================

    st.markdown("""
    <div class="card">

        <div class="card-title">
            🛠️ Technologies Used
        </div>

        <div class="card-text">

            • Python<br>
            • Streamlit<br>
            • Pandas<br>
            • Scikit-learn<br>
            • TF-IDF Vectorization<br>
            • Logistic Regression<br>
            • Joblib<br>
            • Explainable AI

        </div>

    </div>
    """, unsafe_allow_html=True)

    # ==============================
    # XAI
    # ==============================

    st.markdown("""
    <div class="card">

        <div class="card-title">
            🧠 Explainable AI
        </div>

        <div class="card-text">

            TruthLens does not only provide a prediction.

            <br><br>

            It also examines the contribution of words in the
            article and displays words that had stronger positive
            or negative influence on the model's classification.

            <br><br>

            This makes the machine-learning prediction easier
            to inspect and understand.

        </div>

    </div>
    """, unsafe_allow_html=True)

    # ==============================
    # DISCLAIMER
    # ==============================

    st.warning(
        "⚠️ Important: TruthLens is an educational machine-learning "
        "project. A model prediction is not a substitute for "
        "professional fact-checking or verification from reliable sources."
    )

    st.markdown("---")

    st.caption(
        "TruthLens • AI-Powered News Analysis"
    )

