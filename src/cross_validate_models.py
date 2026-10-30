
import os
import json
from pathlib import Path

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

import numpy as np
import xgboost as xgb

from scipy.sparse import csr_matrix

from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

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
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("SentimentModelCrossValidation")
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
# 3. SPARK TF-IDF PIPELINE
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
        stages=[tokenizer, remover, vectorizer, idf]
    )


# --------------------------------------------------
# 4. SPARK SPARSE VECTOR CONVERSION
# --------------------------------------------------

def convert_features(data):

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
            row_values = vector.toArray()
            row_indices = np.flatnonzero(row_values)
            row_values = row_values[row_indices]

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
# 5. XGBOOST CLASS WEIGHTS
# --------------------------------------------------

def get_sample_weights(y):

    classes, counts = np.unique(y, return_counts=True)

    total = len(y)
    number_of_classes = len(classes)

    weights = {
        int(class_id): total / (
            number_of_classes * int(count)
        )
        for class_id, count in zip(classes, counts)
    }

    return np.array([
        weights[int(label)] for label in y
    ])


# --------------------------------------------------
# 6. MODEL DEFINITIONS
# --------------------------------------------------

def create_models():

    return {
        "Naive Bayes": MultinomialNB(
            alpha=1.0
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42
        ),

        "XGBoost": xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=2
        )
    }


# --------------------------------------------------
# 7. MAIN
# --------------------------------------------------

def main():

    print("\n" + "=" * 65)
    print("5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 65)

    df = spark.read.parquet(str(INPUT_PATH))

    df = (
        df.select("cleaned_text", "label")
        .filter(
            col("cleaned_text").isNotNull()
            & (col("cleaned_text") != "")
            & col("label").isin(0, 1, 2)
        )
        .withColumn("label", col("label").cast("double"))
    ).cache()

    print("\nTotal records:", df.count())

    # Same original split as the previous experiments
    train_data, test_data = df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    train_data = train_data.cache()
    test_data = test_data.cache()

    print("Training records:", train_data.count())
    print("Held-out test records:", test_data.count())

    # Fit feature extraction on training data only.
    # The held-out test data is not used in cross-validation.
    print("\nFitting Spark TF-IDF pipeline...")

    pipeline = create_pipeline()
    feature_model = pipeline.fit(train_data)

    train_features = feature_model.transform(train_data)

    X, y = convert_features(train_features)

    print("Feature matrix:", X.shape)

    # --------------------------------------------------
    # 8. STRATIFIED CROSS-VALIDATION
    # --------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    models = create_models()

    results = {}

    for model_name, model_template in models.items():

        print("\n" + "=" * 65)
        print("MODEL:", model_name)
        print("=" * 65)

        fold_results = []

        for fold_number, (train_idx, val_idx) in enumerate(
            cv.split(X, y), start=1
        ):

            print(f"\nFold {fold_number}/5")

            X_fold_train = X[train_idx]
            X_fold_val = X[val_idx]

            y_fold_train = y[train_idx]
            y_fold_val = y[val_idx]

            # Fresh model for each fold
            model = create_models()[model_name]

            if model_name == "XGBoost":

                sample_weights = get_sample_weights(
                    y_fold_train
                )

                model.fit(
                    X_fold_train,
                    y_fold_train,
                    sample_weight=sample_weights,
                    verbose=False
                )

            else:

                model.fit(
                    X_fold_train,
                    y_fold_train
                )

            predictions = model.predict(X_fold_val)

            accuracy = accuracy_score(
                y_fold_val, predictions
            )

            weighted_f1 = f1_score(
                y_fold_val,
                predictions,
                average="weighted",
                zero_division=0
            )

            macro_f1 = f1_score(
                y_fold_val,
                predictions,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0
            )

            weighted_precision = precision_score(
                y_fold_val,
                predictions,
                average="weighted",
                zero_division=0
            )

            weighted_recall = recall_score(
                y_fold_val,
                predictions,
                average="weighted",
                zero_division=0
            )

            fold_result = {
                "accuracy": float(accuracy),
                "weighted_precision": float(weighted_precision),
                "weighted_recall": float(weighted_recall),
                "weighted_f1": float(weighted_f1),
                "macro_f1": float(macro_f1)
            }

            fold_results.append(fold_result)

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Weighted F1: {weighted_f1:.4f}")
            print(f"Macro-F1: {macro_f1:.4f}")

        # --------------------------------------------------
        # 9. MEAN AND STANDARD DEVIATION
        # --------------------------------------------------

        metric_names = [
            "accuracy",
            "weighted_precision",
            "weighted_recall",
            "weighted_f1",
            "macro_f1"
        ]

        summary = {}

        print("\nCROSS-VALIDATION SUMMARY")

        for metric in metric_names:

            values = [
                fold[metric] for fold in fold_results
            ]

            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=1))

            summary[metric] = {
                "mean": mean_value,
                "std": std_value
            }

            print(
                f"{metric}: "
                f"{mean_value:.4f} +/- {std_value:.4f}"
            )

        results[model_name] = {
            "fold_results": fold_results,
            "summary": summary
        }

    # --------------------------------------------------
    # 10. SAVE RESULTS
    # --------------------------------------------------

    report = {
        "cross_validation": "5-fold StratifiedKFold",
        "random_state": 42,
        "training_records": int(len(y)),
        "held_out_test_records": int(test_data.count()),
        "feature_count": int(X.shape[1]),
        "models": results,
        "feature_extraction_note": (
            "TF-IDF vocabulary and IDF were fitted once on the "
            "complete training split before cross-validation. "
            "Therefore, fold validation text influenced unsupervised "
            "feature statistics. The held-out test set was not used."
        )
    }

    output_path = RESULTS_DIR / "cross_validation_results.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 65)
    print("CROSS-VALIDATION COMPLETED")
    print("=" * 65)

    print("Results saved to:", output_path)

    train_data.unpersist()
    test_data.unpersist()
    df.unpersist()


if __name__ == "__main__":

    try:
        main()

    finally:
        spark.stop()
