
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TruthLens - Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #64748b;
    margin-bottom: 25px;
}

.section-title {
    font-size: 26px;
    font-weight: 650;
    margin-top: 20px;
    margin-bottom: 15px;
}

.result-real {
    padding: 25px;
    border-radius: 15px;
    background-color: #dcfce7;
    border: 1px solid #86efac;
    text-align: center;
}

.result-fake {
    padding: 25px;
    border-radius: 15px;
    background-color: #fee2e2;
    border: 1px solid #fca5a5;
    text-align: center;
}

.result-title {
    font-size: 32px;
    font-weight: 700;
}

.result-subtitle {
    font-size: 16px;
    margin-top: 8px;
}

.info-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #e0f2fe;
    border-left: 5px solid #2563eb;
    color: #111827;
    margin: 15px 0;
}

.warning-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #ffedd5;
    border-left: 5px solid #ea580c;
    color: #111827;
    margin: 15px 0;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 30px 0 10px 0;
    font-size: 14px;
}


/* Ensure lower-page content remains visible on light backgrounds */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: #111827 !important;
}

.footer {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    color: #374151 !important;
    padding: 22px 15px 18px 15px;
    margin-top: 30px;
}

.footer br + * {
    color: #374151 !important;
}


/* Global visibility fix */
.stApp, .main, .block-container {
    background-color: #f8fafc !important;
}

[data-testid="stMarkdownContainer"] {
    color: #111827 !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em {
    color: #111827 !important;
}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5,
[data-testid="stMarkdownContainer"] h6 {
    color: #111827 !important;
}

.stTextInput label,
.stTextArea label,
.stSelectbox label,
.stRadio label,
.stButton label {
    color: #111827 !important;
}

div[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #dbe3ef !important;
    border-radius: 14px !important;
}

div[data-testid="stMetric"] * {
    color: #111827 !important;
}

div[data-testid="stDataFrame"] {
    color: #111827 !important;
}

.stAlert {
    color: #111827 !important;
}

footer {
    visibility: hidden;
}

.footer {
    background-color: #ffffff !important;
    border: 1px solid #dbe3ef !important;
    border-radius: 14px !important;
    color: #374151 !important;
    padding: 22px !important;
    margin-top: 30px !important;
    text-align: center !important;
}

.footer * {
    color: #374151 !important;
}


/* Sidebar visibility */
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #374151 !important;
}


/* Prevent lower content from being visually clipped */
.main .block-container {
    min-height: auto !important;
    overflow: visible !important;
    padding-bottom: 5rem !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    metrics = joblib.load("model_metrics.pkl")

    return model, vectorizer, metrics


model, vectorizer, metrics = load_model()


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
# DATASET STATISTICS
# ============================================================
# The original training dataset is kept offline because the CSV files
# are too large for a simple GitHub browser upload. The deployed app
# only needs the trained model/vectorizer/metrics files for prediction.

total_articles = 44898
fake_articles = 23481
real_articles = 21417

# Original training used an 80/20 split
training_size = 35918
testing_size = 8980


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_news(text):

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    probabilities = model.predict_proba(
        text_vector
    )[0]

    return (
        prediction,
        probabilities,
        text_vector
    )


# ============================================================
# EXPLAINABILITY FUNCTION
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

st.sidebar.title("📰 TruthLens")

st.sidebar.markdown(
    "### AI-Powered Fake News Detection"
)

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

st.sidebar.info(
    "TruthLens uses Machine Learning and "
    "Natural Language Processing to classify "
    "news articles as REAL or FAKE."
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">📰 TruthLens</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-Powered Fake News Detection Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # KPI CARDS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Model Accuracy",
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

    # ========================================================
    # NEWS ANALYZER
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🔍 Analyze a News Article'
        '</div>',
        unsafe_allow_html=True
    )

    news_text = st.text_area(
        "Paste the news article below:",
        height=220,
        placeholder=(
            "Paste a complete news article here "
            "and click Analyze News..."
        )
    )

    analyze_button = st.button(
        "🔎 Analyze News",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        if not news_text.strip():

            st.warning(
                "Please enter some news text before analyzing."
            )

        else:

            prediction, probabilities, text_vector = predict_news(
                news_text
            )

            fake_probability = probabilities[0]
            real_probability = probabilities[1]

            if prediction == "REAL":

                st.markdown(
                    """
                    <div class="result-real">
                        <div class="result-title">
                            ✅ REAL NEWS
                        </div>
                        <div class="result-subtitle">
                            The model classified this article
                            as likely REAL.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="result-fake">
                        <div class="result-title">
                            ⚠️ FAKE NEWS
                        </div>
                        <div class="result-subtitle">
                            The model classified this article
                            as likely FAKE.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("### 📊 Prediction Probability")

            prob_df = pd.DataFrame({
                "Class": [
                    "FAKE",
                    "REAL"
                ],
                "Probability": [
                    fake_probability,
                    real_probability
                ]
            })

            fig = px.bar(
                prob_df,
                x="Class",
                y="Probability",
                text_auto=".2%",
                range_y=[0, 1],
                title="Model Confidence"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # =================================================
            # ARTICLE STATISTICS
            # =================================================

            st.markdown("### 📝 Article Statistics")

            word_count = len(
                news_text.split()
            )

            character_count = len(
                news_text
            )

            sentence_count = max(
                1,
                len([
                    x for x in news_text.split(".")
                    if x.strip()
                ])
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Words",
                    word_count
                )

            with col2:

                st.metric(
                    "Characters",
                    character_count
                )

            with col3:

                st.metric(
                    "Sentences",
                    sentence_count
                )

            # =================================================
            # CONFIDENCE
            # =================================================

            confidence = max(
                fake_probability,
                real_probability
            )

            st.markdown("### 🎯 Confidence")

            st.progress(
                float(confidence)
            )

            st.write(
                f"Model confidence: "
                f"**{confidence * 100:.2f}%**"
            )

            if confidence >= 0.90:

                st.success(
                    "The model has high confidence in this prediction."
                )

            elif confidence >= 0.70:

                st.warning(
                    "The model has moderate confidence in this prediction."
                )

            else:

                st.info(
                    "The model has relatively low confidence. "
                    "Consider reviewing the article manually."
                )

            # =================================================
            # EXPLAINABLE AI
            # =================================================

            st.markdown("---")

            st.markdown(
                "### 🧠 Why did the model make this prediction?"
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
                    "#### 🔑 Supporting Signals"
                )

                if not supporting.empty:

                    support_chart = px.bar(
                        supporting.sort_values(
                            "Contribution"
                        ),
                        x="Contribution",
                        y="Word",
                        orientation="h",
                        title="Words supporting prediction"
                    )

                    st.plotly_chart(
                        support_chart,
                        use_container_width=True
                    )

                else:

                    st.info(
                        "No significant supporting words found."
                    )

            with col2:

                st.markdown(
                    "#### ⚖️ Opposing Signals"
                )

                if not opposing.empty:

                    opposing_chart = px.bar(
                        opposing.sort_values(
                            "Contribution"
                        ),
                        x="Contribution",
                        y="Word",
                        orientation="h",
                        title="Words opposing prediction"
                    )

                    st.plotly_chart(
                        opposing_chart,
                        use_container_width=True
                    )

                else:

                    st.info(
                        "No significant opposing words found."
                    )

            st.markdown(
                """
                <div class="warning-box">
                <b>Important:</b> These words are model feature
                signals, not proof that an article is true or false.
                </div>
                """,
                unsafe_allow_html=True
            )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.markdown("---")

    st.markdown(
        """
        <div style="
            background:#e0f2fe;
            border:1px solid #93c5fd;
            border-left:6px solid #2563eb;
            border-radius:14px;
            padding:20px 22px;
            margin:20px 0;
            color:#111827;
        ">
            <div style="
                font-size:18px;
                font-weight:800;
                color:#111827;
                margin-bottom:8px;
            ">
                ⚠️ Disclaimer
            </div>
            <div style="
                font-size:15px;
                line-height:1.65;
                color:#374151;
            ">
                TruthLens is an educational machine learning project.
                Predictions are based on patterns learned from the
                training dataset and should not be treated as definitive
                fact-checking.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# NEWS ANALYZER
# ============================================================

elif page == "🔍 News Analyzer":

    st.markdown(
        '<div class="main-title">🔍 News Analyzer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Analyze individual news articles using the trained ML model.'
        '</div>',
        unsafe_allow_html=True
    )

    news_text = st.text_area(
        "Enter News Article",
        height=300,
        placeholder="Paste news article text here..."
    )

    if st.button(
        "Analyze Article",
        type="primary"
    ):

        if not news_text.strip():

            st.warning(
                "Please enter some text."
            )

        else:

            prediction, probabilities, text_vector = predict_news(
                news_text
            )

            fake_probability = probabilities[0]
            real_probability = probabilities[1]

            if prediction == "REAL":

                st.success(
                    "✅ Prediction: REAL NEWS"
                )

            else:

                st.error(
                    "⚠️ Prediction: FAKE NEWS"
                )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Fake Probability",
                    f"{fake_probability * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Real Probability",
                    f"{real_probability * 100:.2f}%"
                )

            probability_df = pd.DataFrame({
                "Class": [
                    "FAKE",
                    "REAL"
                ],
                "Probability": [
                    fake_probability,
                    real_probability
                ]
            })

            fig = px.pie(
                probability_df,
                names="Class",
                values="Probability",
                hole=0.4,
                title="Prediction Probability"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.markdown("---")

            st.subheader(
                "🧠 Explainable AI"
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
                    "#### 🔑 Supporting Words"
                )

                if not supporting.empty:

                    fig_support = px.bar(
                        supporting.sort_values(
                            "Contribution"
                        ),
                        x="Contribution",
                        y="Word",
                        orientation="h"
                    )

                    st.plotly_chart(
                        fig_support,
                        use_container_width=True
                    )

                else:

                    st.info(
                        "No strong supporting signals found."
                    )

            with col2:

                st.markdown(
                    "#### ⚖️ Opposing Words"
                )

                if not opposing.empty:

                    fig_opposing = px.bar(
                        opposing.sort_values(
                            "Contribution"
                        ),
                        x="Contribution",
                        y="Word",
                        orientation="h"
                    )

                    st.plotly_chart(
                        fig_opposing,
                        use_container_width=True
                    )

                else:

                    st.info(
                        "No strong opposing signals found."
                    )

            st.info(
                "The highlighted words represent model feature "
                "signals and should not be interpreted as factual proof."
            )


# ============================================================
# DATASET ANALYTICS
# ============================================================

elif page == "📈 Dataset Analytics":

    st.markdown(
        '<div class="main-title">📈 Dataset Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore statistics from the dataset used to train the model.'
        '</div>',
        unsafe_allow_html=True
    )

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

    st.markdown("---")

    st.subheader(
        "📊 Class Distribution"
    )

    class_df = pd.DataFrame({
        "Label": [
            "FAKE",
            "REAL"
        ],
        "Count": [
            fake_articles,
            real_articles
        ]
    })

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            class_df,
            names="Label",
            values="Count",
            hole=0.4,
            title="Fake vs Real Articles"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            class_df,
            x="Label",
            y="Count",
            text_auto=True,
            title="Article Count by Class"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader(
        "🧪 Training and Testing Dataset"
    )

    split_df = pd.DataFrame({
        "Dataset": [
            "Training",
            "Testing"
        ],
        "Articles": [
            training_size,
            testing_size
        ]
    })

    fig = px.bar(
        split_df,
        x="Dataset",
        y="Articles",
        text_auto=True,
        title="80/20 Train-Test Split"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader(
        "👀 Dataset Preview"
    )

    st.info(
        "The full CSV dataset is kept offline and is not bundled with "
        "the deployed Streamlit app. The statistics above come from "
        "the dataset used during model training."
    )

    preview_df = pd.DataFrame({
        "Dataset Information": [
            "Total articles",
            "Fake articles",
            "Real articles",
            "Training articles",
            "Testing articles"
        ],
        "Value": [
            f"{total_articles:,}",
            f"{fake_articles:,}",
            f"{real_articles:,}",
            f"{training_size:,}",
            f"{testing_size:,}"
        ]
    })

    st.dataframe(
        preview_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "📊 Model Information":

    st.markdown(
        '<div class="main-title">📊 Model Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine learning model performance and technical details.'
        '</div>',
        unsafe_allow_html=True
    )

    # ========================================================
    # PERFORMANCE METRICS
    # ========================================================

    st.subheader(
        "🎯 Model Performance"
    )

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

    # ========================================================
    # PERFORMANCE CHART
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📊 Performance Comparison"
    )

    performance_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Score": [
            accuracy,
            precision,
            recall,
            f1
        ]
    })

    fig = px.bar(
        performance_df,
        x="Metric",
        y="Score",
        text_auto=".2%",
        range_y=[0, 1],
        title="Model Performance Metrics"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🔢 Confusion Matrix"
    )

    confusion_matrix = metrics.get(
        "confusion_matrix",
        None
    )

    if confusion_matrix is not None:

        cm = pd.DataFrame(
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
            cm,
            use_container_width=True
        )

        st.markdown(
            """
            **How to read the matrix:**

            - **Actual FAKE → Predicted FAKE:** Correctly identified fake news
            - **Actual FAKE → Predicted REAL:** Fake news classified as real
            - **Actual REAL → Predicted FAKE:** Real news classified as fake
            - **Actual REAL → Predicted REAL:** Correctly identified real news
            """
        )

    else:

        st.info(
            "Confusion matrix data was not saved in the model metrics file."
        )

    # ========================================================
    # TRAINING / TESTING
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🧪 Training vs Testing Data"
    )

    split_df = pd.DataFrame({
        "Dataset": [
            "Training",
            "Testing"
        ],
        "Articles": [
            training_size,
            testing_size
        ]
    })

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            split_df,
            names="Dataset",
            values="Articles",
            hole=0.4,
            title="Training / Testing Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.metric(
            "Training Articles",
            f"{training_size:,}"
        )

        st.metric(
            "Testing Articles",
            f"{testing_size:,}"
        )

    # ========================================================
    # ALGORITHM
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🤖 Machine Learning Algorithm"
    )

    st.markdown(
        """
        **Algorithm: Logistic Regression**

        Logistic Regression is a supervised machine learning
        algorithm used for classification problems.

        In TruthLens, it classifies news articles into:

        - 🟢 REAL
        - 🔴 FAKE
        """
    )

    # ========================================================
    # TF-IDF
    # ========================================================

    st.subheader(
        "📝 Text Representation"
    )

    st.markdown(
        """
        **TF-IDF — Term Frequency-Inverse Document Frequency**

        TF-IDF converts text into numerical features that the
        machine learning model can understand.

        The vectorizer used in this project includes:

        - Lowercase conversion
        - English stop-word removal
        - Maximum 50,000 features
        """
    )

    # ========================================================
    # PIPELINE
    # ========================================================

    st.subheader(
        "⚙️ Machine Learning Pipeline"
    )

    st.code(
        """
News Article
     ↓
Text Cleaning
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Prediction
     ↓
REAL / FAKE
        """,
        language="text"
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="main-title">ℹ️ About TruthLens</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'An AI-powered fake news detection project.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        ### 🎯 Project Objective

        TruthLens is a machine learning based application
        designed to classify news articles as REAL or FAKE.

        The project demonstrates how Natural Language Processing
        and Machine Learning can be combined with an interactive
        Streamlit dashboard.

        ### 🧠 Technologies Used

        - Python
        - Pandas
        - Scikit-learn
        - TF-IDF
        - Logistic Regression
        - Streamlit
        - Plotly
        - Joblib

        ### 🔬 Key Features

        - News classification
        - Prediction probabilities
        - Explainable AI
        - Dataset analytics
        - Model performance metrics
        - Confusion matrix
        - Interactive visualizations

        ### 📌 Important Limitation

        The model learns patterns from its training dataset.
        Therefore, a prediction is not equivalent to independent
        journalistic fact-checking or verification of an article.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="
            background:#ffffff;
            border:1px solid #dbe3ef;
            border-left:6px solid #2563eb;
            border-radius:14px;
            padding:22px 24px;
            margin-top:20px;
            color:#111827;
        ">
            <div style="
                font-size:22px;
                font-weight:800;
                color:#111827;
                margin-bottom:8px;
            ">
                TruthLens
            </div>
            <div style="
                font-size:15px;
                color:#374151;
                line-height:1.6;
            ">
                AI-Powered Fake News Detection Dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        TruthLens • AI-Powered Fake News Detection<br>
        Built with Python, Machine Learning & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

