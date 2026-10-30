
# ============================================================
# E-COMMERCE REVIEW INTELLIGENCE
# Big Data Analytics | Sentiment Analysis | Product Insights
# ============================================================

import os
import re
import warnings
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Review Intelligence | E-Commerce Analytics",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5433,
    "database": "ecommerce_sentiment_db",
    "user": "postgres"
}

MODEL_DIR = BASE_DIR / "models"

FEATURE_PIPELINE_PATH = (
    MODEL_DIR / "naive_bayes_feature_pipeline"
)

CLASSIFIER_PATH = (
    MODEL_DIR / "naive_bayes_classifier.joblib"
)

SENTIMENT_COLORS = {
    "Positive": "#329C8C",
    "Neutral": "#E5B65C",
    "Negative": "#D97757"
}

PALETTE = [
    "#765477",
    "#D98255",
    "#329C8C",
    "#D9B66F",
    "#A88BA9",
    "#8B8290",
    "#C7A2A0"
]

# ============================================================
# 2. PROFESSIONAL DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>

@import url(
    'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap'
);

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F7F5F2;
    color: #302833;
}

/* Main content */

.main .block-container {
    max-width: 1500px;
    padding: 2.1rem 2.4rem 3rem 2.4rem;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #342638;
    border-right: 1px solid #49394B;
}

[data-testid="stSidebar"] * {
    color: #F7F2F7;
}

[data-testid="stSidebar"] h1 {
    color: #FFFFFF !important;
    font-size: 1.55rem;
    font-weight: 700;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] hr {
    border-color: #59465A;
}

/* Main headings */

h1, h2, h3 {
    color: #442B45 !important;
    font-weight: 700 !important;
    letter-spacing: -0.4px;
}

h1 {
    font-size: 2rem !important;
}

h2 {
    font-size: 1.45rem !important;
}

h3 {
    font-size: 1.08rem !important;
}

/* Page title */

.page-title {
    color: #442B45;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.8px;
    margin-bottom: 3px;
}

.page-subtitle {
    color: #77717A;
    font-size: 0.96rem;
    margin-bottom: 25px;
}

/* KPI cards */

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E9E1E8;
    border-radius: 13px;
    padding: 20px 21px;
    box-shadow: 0 3px 12px rgba(52, 38, 56, 0.035);
}

[data-testid="stMetricLabel"] {
    color: #746B76 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}

[data-testid="stMetricValue"] {
    color: #442B45 !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

/* Tabs */

.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
    border-bottom: 1px solid #E7DFE7;
}

.stTabs [data-baseweb="tab"] {
    color: #665B68;
    font-size: 0.91rem;
    font-weight: 600;
    padding: 13px 3px;
}

.stTabs [aria-selected="true"] {
    color: #8B526E !important;
}

/* Buttons */

.stButton > button,
.stDownloadButton > button {
    background-color: #442B45;
    color: #FFFFFF;
    border: 1px solid #442B45;
    border-radius: 8px;
    font-weight: 600;
    padding: 9px 17px;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #654765;
    color: #FFFFFF;
    border-color: #654765;
}

/* Inputs */

.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background-color: #FFFFFF;
    color: #302833;
    border-radius: 8px;
    border: 1px solid #DDD4DE;
}

[data-baseweb="select"] > div {
    background-color: #FFFFFF;
    border-color: #DDD4DE;
    border-radius: 8px;
}

/* Tables */

[data-testid="stDataFrame"] {
    border: 1px solid #E8E0E8;
    border-radius: 9px;
}

/* Expanders */

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E8E0E8;
    border-radius: 10px;
}

/* Section labels */

.section-caption {
    color: #817783;
    font-size: 0.88rem;
    margin-top: -8px;
    margin-bottom: 15px;
}

/* Information panels */

.info-panel {
    background: #FFFFFF;
    border: 1px solid #E8E0E8;
    border-left: 4px solid #D98255;
    border-radius: 9px;
    padding: 15px 18px;
    margin: 10px 0 20px 0;
}

/* Prediction result */

.prediction-panel {
    background: #FFFFFF;
    border: 1px solid #E8E0E8;
    border-radius: 12px;
    padding: 22px;
    margin-top: 12px;
}

hr {
    border-color: #E7DFE7;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. DATABASE FUNCTIONS
# ============================================================

def get_connection():
    """Create a PostgreSQL connection."""

    if not st.session_state.get("db_password"):
        raise ValueError("PostgreSQL password is missing.")

    config = DB_CONFIG.copy()
    config["password"] = st.session_state["db_password"]

    return psycopg2.connect(**config)


def fetch_data(query, params=None):
    """Execute a SELECT query and return a DataFrame."""

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params)

            columns = [
                description[0]
                for description in cursor.description
            ]

            rows = cursor.fetchall()

        return pd.DataFrame(rows, columns=columns)

    finally:
        connection.close()


def check_database_connection():
    """Check whether PostgreSQL is reachable."""

    try:
        connection = get_connection()
        connection.close()
        return True, "Connected successfully."

    except Exception as exc:
        return False, str(exc)


# ============================================================
# 4. DATA QUERIES
# ============================================================

def get_overview_metrics():

    query = """
        SELECT
            COUNT(*) AS total_reviews,
            COUNT(DISTINCT clothing_id) AS unique_products,
            ROUND(AVG(rating)::numeric, 2) AS average_rating,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE sentiment_name = 'Positive'
                ) / NULLIF(COUNT(*), 0),
                2
            ) AS positive_percentage

        FROM reviews;
    """

    return fetch_data(query)


def get_sentiment_summary():

    query = """
        SELECT
            sentiment_name,
            COUNT(*) AS total_reviews

        FROM reviews

        GROUP BY sentiment_name

        ORDER BY
            CASE sentiment_name
                WHEN 'Positive' THEN 1
                WHEN 'Neutral' THEN 2
                WHEN 'Negative' THEN 3
            END;
    """

    return fetch_data(query)


def get_rating_summary():

    query = """
        SELECT
            rating,
            COUNT(*) AS total_reviews

        FROM reviews

        GROUP BY rating

        ORDER BY rating;
    """

    return fetch_data(query)


def get_department_summary():

    query = """
        SELECT
            department_name,
            COUNT(*) AS total_reviews,

            ROUND(AVG(rating)::numeric, 2)
                AS average_rating,

            COUNT(*) FILTER (
                WHERE sentiment_name = 'Positive'
            ) AS positive_reviews,

            COUNT(*) FILTER (
                WHERE sentiment_name = 'Neutral'
            ) AS neutral_reviews,

            COUNT(*) FILTER (
                WHERE sentiment_name = 'Negative'
            ) AS negative_reviews,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE sentiment_name = 'Negative'
                ) / NULLIF(COUNT(*), 0),
                2
            ) AS negative_percentage

        FROM reviews

        GROUP BY department_name

        ORDER BY total_reviews DESC;
    """

    return fetch_data(query)


def get_department_sentiment():

    query = """
        SELECT
            department_name,
            sentiment_name,
            COUNT(*) AS total_reviews

        FROM reviews

        GROUP BY
            department_name,
            sentiment_name

        ORDER BY department_name;
    """

    return fetch_data(query)


def get_product_summary(min_reviews=20):

    query = """
        SELECT
            clothing_id,
            COUNT(*) AS total_reviews,

            ROUND(AVG(rating)::numeric, 2)
                AS average_rating,

            COUNT(*) FILTER (
                WHERE sentiment_name = 'Positive'
            ) AS positive_reviews,

            COUNT(*) FILTER (
                WHERE sentiment_name = 'Neutral'
            ) AS neutral_reviews,

            COUNT(*) FILTER (
                WHERE sentiment_name = 'Negative'
            ) AS negative_reviews,

            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE sentiment_name = 'Negative'
                ) / NULLIF(COUNT(*), 0),
                2
            ) AS negative_percentage

        FROM reviews

        GROUP BY clothing_id

        HAVING COUNT(*) >= %s

        ORDER BY average_rating DESC, total_reviews DESC;
    """

    return fetch_data(query, (min_reviews,))


def get_review_data(
    sentiment=None,
    department=None,
    limit=300
):

    query = """
        SELECT
            review_id,
            clothing_id,
            title,
            review_text,
            rating,
            sentiment_name,
            department_name,
            class_name,
            positive_feedback_count

        FROM reviews

        WHERE
            (%s IS NULL OR sentiment_name = %s)
            AND
            (%s IS NULL OR department_name = %s)

        ORDER BY positive_feedback_count DESC NULLS LAST

        LIMIT %s;
    """

    return fetch_data(
        query,
        (
            sentiment,
            sentiment,
            department,
            department,
            limit
        )
    )


# ============================================================
# 5. PLOTLY DESIGN HELPERS
# ============================================================

def style_figure(fig, title=None, height=390):

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",

        font=dict(
            family="DM Sans, sans-serif",
            color="#443D47",
            size=12
        ),

        title=dict(
            text=title,
            font=dict(
                size=17,
                color="#442B45"
            ),
            x=0.02,
            xanchor="left"
        ),

        height=height,

        margin=dict(
            l=35,
            r=25,
            t=65,
            b=45
        ),

        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            borderwidth=0
        ),

        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font_color="#302833"
        )
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#E8E0E8",
        tickfont=dict(color="#716A75")
    )

    fig.update_yaxes(
        gridcolor="#EEE9EF",
        zerolinecolor="#E8E0E8",
        tickfont=dict(color="#716A75")
    )

    return fig


def empty_message(message):

    st.info(message)


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("Review Intelligence")

    st.caption("E-COMMERCE SENTIMENT ANALYTICS")

    st.divider()

    st.subheader("Database Connection")

    password = st.text_input(
        "PostgreSQL password",
        type="password",
        key="password_input"
    )

    if st.button(
        "Connect to Database",
        use_container_width=True
    ):

        st.session_state["db_password"] = password

        success, message = check_database_connection()

        if success:
            st.session_state["db_connected"] = True
            st.success("Database connected.")

        else:
            st.session_state["db_connected"] = False
            st.error("Connection failed.")
            st.caption(message)

    if st.session_state.get("db_connected"):

        st.success("PostgreSQL connected")

    st.divider()

    st.caption("PROJECT TECHNOLOGY")

    st.markdown("""
    **Processing:** Apache Spark

    **Database:** PostgreSQL

    **Machine Learning:** Naive Bayes

    **Interface:** Streamlit

    **Visualisation:** Plotly
    """)

    st.divider()

    st.caption("E-Commerce Review Intelligence")


# ============================================================
# 7. CONNECTION CHECK
# ============================================================

if not st.session_state.get("db_connected"):

    st.markdown(
        '<div class="page-title">Review Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'E-Commerce sentiment analysis and product insights'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-panel">
        <b>Database connection required</b><br>
        Enter your PostgreSQL password in the sidebar and
        connect to load the analytics dashboard.
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ============================================================
# 8. LOAD DASHBOARD DATA
# ============================================================

try:

    overview = get_overview_metrics().iloc[0]

    sentiment_df = get_sentiment_summary()

    rating_df = get_rating_summary()

    department_df = get_department_summary()

    department_sentiment_df = get_department_sentiment()

except Exception as exc:

    st.error("Unable to load dashboard data.")

    st.exception(exc)

    st.stop()


# ============================================================
# 9. PAGE HEADER
# ============================================================

st.markdown(
    '<div class="page-title">E-Commerce Review Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-subtitle">'
    'Big Data Analytics  |  Customer Sentiment  |  Product Insights'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# 10. KPI CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Reviews",
        f"{int(overview['total_reviews']):,}"
    )

with col2:

    st.metric(
        "Unique Products",
        f"{int(overview['unique_products']):,}"
    )

with col3:

    st.metric(
        "Average Rating",
        f"{float(overview['average_rating']):.2f} / 5"
    )

with col4:

    st.metric(
        "Positive Reviews",
        f"{float(overview['positive_percentage']):.2f}%"
    )


st.write("")


# ============================================================
# 11. DASHBOARD TABS
# ============================================================

tab_overview, tab_sentiment, tab_department, tab_product, tab_prediction = (
    st.tabs([
        "Overview",
        "Sentiment Analysis",
        "Department Analysis",
        "Product Analysis",
        "Live Prediction"
    ])
)


# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with tab_overview:

    st.header("Business Overview")

    st.caption(
        "A consolidated view of customer reviews and rating patterns."
    )

    left, right = st.columns(2, gap="large")

    # Sentiment distribution

    with left:

        sentiment_order = [
            "Positive",
            "Neutral",
            "Negative"
        ]

        sentiment_df["sentiment_name"] = pd.Categorical(
            sentiment_df["sentiment_name"],
            categories=sentiment_order,
            ordered=True
        )

        sentiment_df = sentiment_df.sort_values(
            "sentiment_name"
        )

        fig = px.pie(
            sentiment_df,
            names="sentiment_name",
            values="total_reviews",
            hole=0.62,
            color="sentiment_name",
            color_discrete_map=SENTIMENT_COLORS
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent",
            marker=dict(
                line=dict(
                    color="#FFFFFF",
                    width=3
                )
            )
        )

        style_figure(
            fig,
            "Customer Sentiment Distribution",
            410
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            theme=None
        )

    # Rating distribution

    with right:

        fig = px.bar(
            rating_df,
            x="rating",
            y="total_reviews",
            text="total_reviews"
        )

        fig.update_traces(
            marker_color="#765477",
            marker_line_color="#FFFFFF",
            marker_line_width=1.5,
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title="Customer Rating",
            yaxis_title="Number of Reviews",
            bargap=0.35
        )

        style_figure(
            fig,
            "Review Rating Distribution",
            410
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            theme=None
        )

    st.divider()

    st.subheader("Department Performance")

    if not department_df.empty:

        fig = px.bar(
            department_df.sort_values("total_reviews"),
            x="total_reviews",
            y="department_name",
            orientation="h",
            text="total_reviews",
            color="average_rating",
            color_continuous_scale=[
                "#E8DDE8",
                "#765477",
                "#442B45"
            ]
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title="Total Reviews",
            yaxis_title="",
            coloraxis_colorbar_title="Average Rating"
        )

        style_figure(
            fig,
            "Review Volume by Department",
            390
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            theme=None
        )

    st.caption(
        "Product identifiers represent Clothing IDs from the source dataset."
    )


# ============================================================
# TAB 2: SENTIMENT ANALYSIS
# ============================================================

with tab_sentiment:

    st.header("Sentiment Analysis")

    st.caption(
        "Explore the distribution of positive, neutral and negative reviews."
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:

        fig = px.bar(
            sentiment_df,
            x="sentiment_name",
            y="total_reviews",
            text="total_reviews",
            color="sentiment_name",
            color_discrete_map=SENTIMENT_COLORS
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title="Sentiment",
            yaxis_title="Number of Reviews",
            showlegend=False
        )

        style_figure(
            fig,
            "Sentiment Comparison",
            390
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            theme=None
        )

    with col2:

        fig = px.pie(
            sentiment_df,
            names="sentiment_name",
            values="total_reviews",
            hole=0.55,
            color="sentiment_name",
            color_discrete_map=SENTIMENT_COLORS
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        style_figure(
            fig,
            "Sentiment Share",
            390
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            theme=None
        )

    st.divider()

    st.subheader("Explore Customer Reviews")

    filter1, filter2 = st.columns(2)

    with filter1:

        selected_sentiment = st.selectbox(
            "Filter by sentiment",
            [
                "All",
                "Positive",
                "Neutral",
                "Negative"
            ]
        )

    with filter2:

        departments = [
            "All"
        ] + sorted(
            department_df["department_name"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_department = st.selectbox(
            "Filter by department",
            departments
        )

    sentiment_filter = (
        None if selected_sentiment == "All"
        else selected_sentiment
    )

    department_filter = (
        None if selected_department == "All"
        else selected_department
    )

    review_df = get_review_data(
        sentiment=sentiment_filter,
        department=department_filter,
        limit=300
    )

    if not review_df.empty:

        st.dataframe(
            review_df,
            use_container_width=True,
            hide_index=True,
            height=350
        )

        csv = review_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Filtered Reviews",
            data=csv,
            file_name="filtered_reviews.csv",
            mime="text/csv"
        )

    else:

        empty_message("No reviews match the selected filters.")


# ============================================================
# TAB 3: DEPARTMENT ANALYSIS
# ============================================================

with tab_department:

    st.header("Department Analysis")

    st.caption(
        "Compare review volume, average ratings and negative feedback."
    )

    if not department_df.empty:

        col1, col2 = st.columns(2, gap="large")

        # Average rating

        with col1:

            fig = px.bar(
                department_df.sort_values("average_rating"),
                x="average_rating",
                y="department_name",
                orientation="h",
                text="average_rating"
            )

            fig.update_traces(
                marker_color="#765477",
                texttemplate="%{text:.2f}",
                textposition="outside"
            )

            fig.update_layout(
                xaxis_title="Average Rating",
                yaxis_title="",
                xaxis_range=[0, 5.5]
            )

            style_figure(
                fig,
                "Average Rating by Department",
                390
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                theme=None
            )

        # Negative review percentage

        with col2:

            fig = px.bar(
                department_df.sort_values("negative_percentage"),
                x="negative_percentage",
                y="department_name",
                orientation="h",
                text="negative_percentage"
            )

            fig.update_traces(
                marker_color="#D97757",
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig.update_layout(
                xaxis_title="Negative Reviews (%)",
                yaxis_title=""
            )

            style_figure(
                fig,
                "Negative Review Percentage",
                390
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                theme=None
            )

        st.divider()

        st.subheader("Sentiment by Department")

        fig = px.bar(
            department_sentiment_df,
            x="department_name",
            y="total_reviews",
            color="sentiment_name",
            color_discrete_map=SENTIMENT_COLORS,
            barmode="stack",
            category_orders={
                "sentiment_name": [
                    "Positive",
                    "Neutral",
                    "Negative"
                ]
            }
        )

        fig.update_layout(
            xaxis_title="Department",
            yaxis_title="Number of Reviews",
            legend_title="Sentiment"
        )

        style_figure(
            fig,
            "Department Sentiment Breakdown",
            430
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            theme=None
        )

        st.divider()

        st.subheader("Department Summary")

        display_department = department_df.rename(
            columns={
                "department_name": "Department",
                "total_reviews": "Total Reviews",
                "average_rating": "Average Rating",
                "positive_reviews": "Positive Reviews",
                "neutral_reviews": "Neutral Reviews",
                "negative_reviews": "Negative Reviews",
                "negative_percentage": "Negative (%)"
            }
        )

        st.dataframe(
            display_department,
            use_container_width=True,
            hide_index=True
        )

    else:

        empty_message("Department data is unavailable.")


# ============================================================
# TAB 4: PRODUCT ANALYSIS
# ============================================================

with tab_product:

    st.header("Product Analysis")

    st.caption(
        "Identify highly rated products and products receiving "
        "comparatively more negative feedback."
    )

    min_reviews = st.slider(
        "Minimum number of reviews per product",
        min_value=1,
        max_value=100,
        value=20,
        step=1
    )

    product_df = get_product_summary(
        min_reviews=min_reviews
    )

    if not product_df.empty:

        best_products = product_df.sort_values(
            ["average_rating", "total_reviews"],
            ascending=[False, False]
        ).head(10)

        negative_products = product_df.sort_values(
            ["negative_percentage", "total_reviews"],
            ascending=[False, False]
        ).head(10)

        col1, col2 = st.columns(2, gap="large")

        with col1:

            st.subheader("Highest Rated Products")

            fig = px.bar(
                best_products.sort_values("average_rating"),
                x="average_rating",
                y="clothing_id",
                orientation="h",
                text="average_rating"
            )

            fig.update_traces(
                marker_color="#329C8C",
                texttemplate="%{text:.2f}",
                textposition="outside"
            )

            fig.update_layout(
                xaxis_title="Average Rating",
                yaxis_title="Clothing ID",
                xaxis_range=[0, 5.5]
            )

            style_figure(
                fig,
                "Top Rated Products",
                430
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                theme=None
            )

        with col2:

            st.subheader("Products with More Negative Feedback")

            fig = px.bar(
                negative_products.sort_values(
                    "negative_percentage"
                ),
                x="negative_percentage",
                y="clothing_id",
                orientation="h",
                text="negative_percentage"
            )

            fig.update_traces(
                marker_color="#D97757",
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig.update_layout(
                xaxis_title="Negative Reviews (%)",
                yaxis_title="Clothing ID"
            )

            style_figure(
                fig,
                "Negative Feedback by Product",
                430
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                theme=None
            )

        st.divider()

        st.subheader("Product Performance Table")

        st.caption(
            "Products are identified by Clothing ID; the dataset "
            "does not provide product names."
        )

        st.dataframe(
            product_df,
            use_container_width=True,
            hide_index=True,
            height=380
        )

        csv = product_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Product Analysis",
            data=csv,
            file_name="product_analysis.csv",
            mime="text/csv"
        )

    else:

        empty_message(
            "No products meet the selected minimum-review threshold."
        )


# ============================================================
# TAB 5: LIVE SENTIMENT PREDICTION
# ============================================================

with tab_prediction:

    st.header("Live Sentiment Prediction")

    st.caption(
        "Enter a customer review to predict its sentiment "
        "using the trained Naive Bayes model."
    )

    st.markdown("""
    <div class="info-panel">
        <b>Prediction model</b><br>
        Multinomial Naive Bayes with a saved Spark TF-IDF
        feature pipeline.
    </div>
    """, unsafe_allow_html=True)

    review_title = st.text_input(
        "Review title",
        placeholder="Example: Excellent quality and fit"
    )

    review_text = st.text_area(
        "Customer review",
        placeholder=(
            "Enter the customer's review here..."
        ),
        height=160
    )

    predict_button = st.button(
        "Analyse Review",
        use_container_width=True
    )

    if predict_button:

        combined_text = (
            str(review_title or "") + " " +
            str(review_text or "")
        ).strip()

        if not combined_text:

            st.warning("Please enter a review before predicting.")

        else:

            try:

                # --------------------------------------------
                # Clean text in the same manner as preprocessing
                # --------------------------------------------

                cleaned_text = combined_text.lower()

                cleaned_text = re.sub(
                    r"[^\w\s]",
                    " ",
                    cleaned_text,
                    flags=re.UNICODE
                )

                cleaned_text = re.sub(
                    r"\s+",
                    " ",
                    cleaned_text
                ).strip()

                if not cleaned_text:

                    st.warning(
                        "The review does not contain usable text."
                    )

                    st.stop()

                # --------------------------------------------
                # Check model files
                # --------------------------------------------

                if not FEATURE_PIPELINE_PATH.exists():

                    raise FileNotFoundError(
                        "Saved Spark feature pipeline not found."
                    )

                if not CLASSIFIER_PATH.exists():

                    raise FileNotFoundError(
                        "Saved Naive Bayes classifier not found."
                    )

                # --------------------------------------------
                # Load saved classifier
                # --------------------------------------------

                classifier = joblib.load(
                    CLASSIFIER_PATH
                )

                # --------------------------------------------
                # Start Spark only when prediction is requested
                # --------------------------------------------

                from pyspark.sql import SparkSession
                from pyspark.ml import PipelineModel
                from scipy.sparse import csr_matrix

                os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
                os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

                @st.cache_resource
                def get_spark_session():

                    spark = (
                        SparkSession.builder
                        .appName("ReviewIntelligencePrediction")
                        .master("local[2]")
                        .config(
                            "spark.driver.host",
                            "127.0.0.1"
                        )
                        .config(
                            "spark.driver.bindAddress",
                            "127.0.0.1"
                        )
                        .config(
                            "spark.hadoop.fs.defaultFS",
                            "file:///"
                        )
                        .config(
                            "spark.hadoop.fs.default.name",
                            "file:///"
                        )
                        .config(
                            "spark.sql.shuffle.partitions",
                            "4"
                        )
                        .getOrCreate()
                    )

                    spark.sparkContext.setLogLevel("ERROR")

                    return spark

                @st.cache_resource
                def load_feature_pipeline(path):

                    return PipelineModel.load(path)

                spark = get_spark_session()

                feature_pipeline = load_feature_pipeline(
                    str(FEATURE_PIPELINE_PATH)
                )

                # --------------------------------------------
                # Transform input text with saved Spark pipeline
                # --------------------------------------------

                input_df = spark.createDataFrame(
                    [(cleaned_text,)],
                    ["cleaned_text"]
                )

                transformed_df = feature_pipeline.transform(
                    input_df
                )

                feature_vector = (
                    transformed_df
                    .select("features")
                    .first()["features"]
                )

                feature_array = feature_vector.toArray()

                feature_matrix = csr_matrix(
                    feature_array.reshape(1, -1)
                )

                # --------------------------------------------
                # Predict sentiment
                # --------------------------------------------

                predicted_label = int(
                    classifier.predict(feature_matrix)[0]
                )

                probabilities = classifier.predict_proba(
                    feature_matrix
                )[0]

                sentiment_labels = {
                    0: "Negative",
                    1: "Neutral",
                    2: "Positive"
                }

                predicted_sentiment = sentiment_labels[
                    predicted_label
                ]

                confidence = float(
                    probabilities[predicted_label] * 100
                )

                # --------------------------------------------
                # Display result
                # --------------------------------------------

                st.divider()

                st.subheader("Prediction Result")

                result_color = SENTIMENT_COLORS[
                    predicted_sentiment
                ]

                st.markdown(
                    f"""
                    <div class="prediction-panel">
                        <div style="
                            color:#817783;
                            font-size:0.9rem;
                            font-weight:600;
                        ">
                            PREDICTED SENTIMENT
                        </div>

                        <div style="
                            color:{result_color};
                            font-size:2rem;
                            font-weight:700;
                            margin-top:8px;
                        ">
                            {predicted_sentiment}
                        </div>

                        <div style="
                            color:#746B76;
                            margin-top:7px;
                        ">
                            Model confidence: {confidence:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                probability_df = pd.DataFrame({
                    "Sentiment": [
                        "Negative",
                        "Neutral",
                        "Positive"
                    ],
                    "Probability": [
                        float(probabilities[0]) * 100,
                        float(probabilities[1]) * 100,
                        float(probabilities[2]) * 100
                    ]
                })

                fig = px.bar(
                    probability_df,
                    x="Sentiment",
                    y="Probability",
                    text="Probability",
                    color="Sentiment",
                    color_discrete_map=SENTIMENT_COLORS
                )

                fig.update_traces(
                    texttemplate="%{text:.2f}%",
                    textposition="outside"
                )

                fig.update_layout(
                    yaxis_title="Probability (%)",
                    xaxis_title="",
                    yaxis_range=[
                        0,
                        max(
                            105,
                            probability_df["Probability"].max() + 10
                        )
                    ],
                    showlegend=False
                )

                style_figure(
                    fig,
                    "Sentiment Probability Distribution",
                    350
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    theme=None
                )

                st.caption(
                    "The prediction is generated by the trained "
                    "model. Confidence represents the model's "
                    "predicted class probability."
                )

            except Exception as exc:

                st.error(
                    "Prediction could not be completed."
                )

                st.exception(exc)


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

footer_left, footer_right = st.columns(2)

with footer_left:

    st.caption(
        "E-Commerce Review Intelligence | Big Data Analytics"
    )

with footer_right:

    st.caption(
        "Apache Spark  |  PostgreSQL  |  Streamlit"
    )
