
import os
import json
from pathlib import Path

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

import numpy as np
import joblib

from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF
)


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data/processed/reviews.parquet"
MODELS_DIR = BASE_DIR / "models"

FEATURE_PIPELINE_PATH = MODELS_DIR / "naive_bayes_feature_pipeline"
CLASSIFIER_PATH = MODELS_DIR / "naive_bayes_classifier.joblib"
METADATA_PATH = MODELS_DIR / "naive_bayes_metadata.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("FinalNaiveBayesTraining")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .config("spark.hadoop.fs.default.name", "file:///")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# --------------------------------------------------
# 3. TF-IDF PIPELINE
# --------------------------------------------------

def create_pipeline():

    tokenizer = RegexTokenizer(
        inputCol="cleaned_text",
        outputCol="tokens",
        pattern="\\W+",
        minTokenLength=2
    )

    remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="filtered_tokens"
    )

    vectorizer = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="raw_features",
        vocabSize=20000,
        minDF=2.0
    )

    idf = IDF(
        inputCol="raw_features",
        outputCol="features"
    )

    return Pipeline(
        stages=[
            tokenizer,
            remover,
            vectorizer,
            idf
        ]
    )


# --------------------------------------------------
# 4. SPARK FEATURES TO SCIPY SPARSE MATRIX
# --------------------------------------------------

def convert_to_sparse_matrix(data):

    rows = data.select("features", "label").collect()

    values = []
    indices = []
    indptr = [0]
    labels = []

    feature_count = None

    for row in rows:

        vector = row["features"]

        if feature_count is None:
            feature_count = vector.size

        if hasattr(vector, "indices"):
            row_indices = vector.indices
            row_values = vector.values
        else:
            dense_values = vector.toArray()
            row_indices = np.flatnonzero(dense_values)
            row_values = dense_values[row_indices]

        indices.extend(row_indices.tolist())
        values.extend(row_values.tolist())
        indptr.append(len(values))

        labels.append(int(row["label"]))

    matrix = csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32)
        ),
        shape=(len(rows), feature_count),
        dtype=np.float32
    )

    return matrix, np.asarray(labels, dtype=np.int32)


# --------------------------------------------------
# 5. TRAIN FINAL MODEL
# --------------------------------------------------

def main():

    print("\n" + "=" * 60)
    print("FINAL NAIVE BAYES MODEL TRAINING")
    print("=" * 60)

    # Read processed dataset
    df = spark.read.parquet(str(INPUT_PATH))

    df = (
        df.select("cleaned_text", "label")
        .filter(
            col("cleaned_text").isNotNull()
            & (col("cleaned_text") != "")
            & col("label").isin(0, 1, 2)
        )
        .cache()
    )

    total_records = df.count()

    print("\nTotal training records:", total_records)

    # Fit TF-IDF on the complete processed dataset
    print("\nFitting the TF-IDF feature pipeline...")

    pipeline = create_pipeline()
    feature_model = pipeline.fit(df)

    transformed_data = feature_model.transform(df)

    X, y = convert_to_sparse_matrix(transformed_data)

    print("Feature matrix shape:", X.shape)
    print("Label distribution:")

    unique_labels, counts = np.unique(y, return_counts=True)

    label_names = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }

    for label, count in zip(unique_labels, counts):
        print(
            f"{label_names[int(label)]}: {count}"
        )

    # Train Naive Bayes
    print("\nTraining Multinomial Naive Bayes...")

    classifier = MultinomialNB(alpha=1.0)
    classifier.fit(X, y)

    print("\nTraining completed!")

    # --------------------------------------------------
    # 6. SAVE ARTIFACTS
    # --------------------------------------------------

    print("\nSaving feature pipeline...")

    feature_model.write().overwrite().save(
        str(FEATURE_PIPELINE_PATH)
    )

    print("Feature pipeline saved.")

    print("\nSaving classifier...")

    joblib.dump(
        classifier,
        CLASSIFIER_PATH
    )

    print("Classifier saved.")

    # Save metadata
    metadata = {
        "model_name": "Multinomial Naive Bayes",
        "algorithm": "MultinomialNB",
        "alpha": 1.0,
        "training_records": int(total_records),
        "feature_count": int(X.shape[1]),
        "label_mapping": {
            "0": "Negative",
            "1": "Neutral",
            "2": "Positive"
        },
        "cross_validation": {
            "method": "Strict 5-fold Stratified CV",
            "macro_f1_mean": 0.6002,
            "macro_f1_std": 0.0144,
            "weighted_f1_mean": 0.7958,
            "weighted_f1_std": 0.0089,
            "accuracy_mean": 0.7813,
            "accuracy_std": 0.0098
        }
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nMetadata saved.")

    print("\n" + "=" * 60)
    print("FINAL MODEL ARTIFACTS CREATED SUCCESSFULLY")
    print("=" * 60)

    print("\nSaved files:")
    print(FEATURE_PIPELINE_PATH)
    print(CLASSIFIER_PATH)
    print(METADATA_PATH)

    df.unpersist()


if __name__ == "__main__":

    try:
        main()

    finally:
        spark.stop()
