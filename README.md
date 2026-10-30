# E-Commerce Review Intelligence

**Big Data Analytics | Customer Sentiment Analysis | Product Insights**

## Overview

E-Commerce Review Intelligence is a data analytics project that processes customer reviews, classifies sentiment, and generates product and department-level insights.

It integrates PySpark, machine learning, PostgreSQL, and Streamlit into an end-to-end analytics application.

## Technology Stack

- Python
- Apache Spark and PySpark
- Spark ML
- TF-IDF
- Multinomial Naive Bayes
- Logistic Regression
- XGBoost
- PostgreSQL
- Streamlit
- Plotly

## Key Features

- Review preprocessing and text cleaning
- Three-class sentiment classification
- TF-IDF feature extraction
- Machine learning model comparison
- Stratified 5-fold cross-validation
- PostgreSQL storage and SQL analytics
- Department-level sentiment analysis
- Product-level review insights
- Interactive sentiment prediction dashboard

## Dataset

Women's E-Commerce Clothing Reviews dataset.

[Kaggle Dataset](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

Download the CSV separately and place it inside `data/raw/`.

The dataset is not included in this repository.

## Sentiment Labelling

Sentiment is derived from review ratings:

| Rating | Sentiment |
|---|---|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

These are rating-derived labels, so the task uses weak supervision rather than independently annotated sentiment labels.

## Verified Dataset Summary

- Processed reviews: 19,759
- Unique products: 1,118
- Average rating: 4.18/5

## Model Evaluation

The current held-out test results for Multinomial Naive Bayes are:

- Accuracy: 77.99%
- Weighted F1-score: 0.7961
- Macro-F1: 0.6061

Strict stratified 5-fold cross-validation:

- Macro-F1: 0.6083 ± 0.0073

The cross-validation result is reported separately from the held-out test result.

## Running the Project

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Download the dataset and place it in `data/raw/`.
5. Configure PostgreSQL and the application database connection.
6. Run preprocessing:

   `python src/preprocess_data.py`

7. Train and evaluate models:

   `python src/train_models.py`

8. Train the final model:

   `python src/train_final_model.py`

9. Import processed reviews into PostgreSQL:

   `python src/load_reviews_to_postgres.py`

10. Launch the dashboard:

   `streamlit run dashboard/app.py`

## Project Scope

The project uses a local PySpark environment to demonstrate Spark-based data processing and machine learning workflows. It does not claim to use a multi-node Spark cluster.