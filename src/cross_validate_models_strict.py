
import os
import gc
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
from pyspark.sql.functions import col, monotonically_increasing_id
from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    StringType,
    DoubleType
)
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
    .appName("StrictSentimentCrossValidation")
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
# 3. FEATURE PIPELINE
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
# 4. SPARSE MATRIX CONVERSION
# --------------------------------------------------

def convert_features(data):

    rows = (
        data.select("row_id", "features", "label")
        .orderBy("row_id")
        .collect()
    )

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
# 6. MODEL CREATION
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
# 7. SPARK DATAFRAME SCHEMA FOR EACH FOLD
# --------------------------------------------------

fold_schema = StructType([
    StructField("row_id", LongType(), False),
    StructField("cleaned_text", StringType(), False),
    StructField("label", DoubleType(), False)
])


def make_fold_dataframe(rows, indices):

    selected_rows = [
        rows[int(index)]
        for index in indices
    ]

    return spark.createDataFrame(
        selected_rows,
        schema=fold_schema
    )


# --------------------------------------------------
# 8. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\n" + "=" * 65)
    print("STRICT 5-FOLD STRATIFIED CROSS-VALIDATION")
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
        .withColumn("row_id", monotonically_increasing_id())
    ).cache()

    print("\nTotal records:", df.count())

    # Same original train-test split
    train_data, test_data = df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    train_data = train_data.cache()
    test_data = test_data.cache()

    print("Training records:", train_data.count())
    print("Held-out test records:", test_data.count())

    # Collect only the training text and labels.
    # The held-out test set is never used in CV.
    training_rows = (
        train_data.select("row_id", "cleaned_text", "label")
        .orderBy("row_id")
        .collect()
    )

    rows = [
        (
            int(row["row_id"]),
            row["cleaned_text"],
            float(row["label"])
        )
        for row in training_rows
    ]

    y = np.asarray(
        [int(row[2]) for row in rows],
        dtype=np.int32
    )

    print("\nPreparing stratified folds...")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    model_names = [
        "Naive Bayes",
        "Logistic Regression",
        "XGBoost"
    ]

    results = {}

    # --------------------------------------------------
    # 9. FIVE FOLDS
    # --------------------------------------------------

    for fold_number, (train_idx, val_idx) in enumerate(
        cv.split(np.zeros(len(y)), y),
        start=1
    ):

        print("\n" + "=" * 65)
        print(f"FOLD {fold_number}/5")
        print("=" * 65)

        # Create independent Spark dataframes
        fold_train = make_fold_dataframe(rows, train_idx)
        fold_validation = make_fold_dataframe(rows, val_idx)

        print("Fold training records:", fold_train.count())
        print("Fold validation records:", fold_validation.count())

        # Fit vocabulary and IDF ONLY on fold training data
        print("\nFitting TF-IDF on fold training data...")

        pipeline = create_pipeline()
        feature_model = pipeline.fit(fold_train)

        train_features = feature_model.transform(fold_train)
        validation_features = feature_model.transform(
            fold_validation
        )

        X_train, y_train = convert_features(train_features)
        X_val, y_val = convert_features(validation_features)

        print("Training matrix:", X_train.shape)
        print("Validation matrix:", X_val.shape)

        # --------------------------------------------------
        # 10. TRAIN AND EVALUATE EACH MODEL
        # --------------------------------------------------

        for model_name in model_names:

            print("\nModel:", model_name)

            model = create_models()[model_name]

            if model_name == "XGBoost":

                sample_weights = get_sample_weights(y_train)

                model.fit(
                    X_train,
                    y_train,
                    sample_weight=sample_weights,
                    verbose=False
                )

            else:

                model.fit(X_train, y_train)

            predictions = model.predict(X_val)

            fold_metrics = {
                "accuracy": float(
                    accuracy_score(y_val, predictions)
                ),
                "weighted_precision": float(
                    precision_score(
                        y_val, predictions,
                        average="weighted",
                        zero_division=0
                    )
                ),
                "weighted_recall": float(
                    recall_score(
                        y_val, predictions,
                        average="weighted",
                        zero_division=0
                    )
                ),
                "weighted_f1": float(
                    f1_score(
                        y_val, predictions,
                        average="weighted",
                        zero_division=0
                    )
                ),
                "macro_f1": float(
                    f1_score(
                        y_val, predictions,
                        labels=[0, 1, 2],
                        average="macro",
                        zero_division=0
                    )
                )
            }

            results.setdefault(model_name, []).append(
                fold_metrics
            )

            print(f"Accuracy: {fold_metrics['accuracy']:.4f}")
            print(
                f"Weighted F1: {fold_metrics['weighted_f1']:.4f}"
            )
            print(f"Macro-F1: {fold_metrics['macro_f1']:.4f}")

            del model
            gc.collect()

        # Release fold resources
        del feature_model
        del train_features
        del validation_features
        del X_train
        del X_val
        del y_train
        del y_val
        del fold_train
        del fold_validation
        gc.collect()

    # --------------------------------------------------
    # 11. CALCULATE MEAN AND STANDARD DEVIATION
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL CROSS-VALIDATION SUMMARY")
    print("=" * 65)

    metric_names = [
        "accuracy",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "macro_f1"
    ]

    final_results = {}

    for model_name in model_names:

        print("\nMODEL:", model_name)

        model_summary = {}

        for metric in metric_names:

            values = [
                fold[metric]
                for fold in results[model_name]
            ]

            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=1))

            model_summary[metric] = {
                "mean": mean_value,
                "std": std_value
            }

            print(
                f"{metric}: "
                f"{mean_value:.4f} +/- {std_value:.4f}"
            )

        final_results[model_name] = {
            "fold_results": results[model_name],
            "summary": model_summary
        }

    # --------------------------------------------------
    # 12. SAVE RESULTS
    # --------------------------------------------------

    report = {
        "method": "Strict 5-fold StratifiedKFold",
        "random_state": 42,
        "training_records": len(rows),
        "held_out_test_records": int(test_data.count()),
        "feature_extraction": (
            "Tokenizer, vocabulary, and IDF were fitted "
            "independently inside each fold using only its "
            "training partition."
        ),
        "models": final_results
    }

    output_path = (
        RESULTS_DIR / "strict_cross_validation_results.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nResults saved to:", output_path)
    print("\nSTRICT CROSS-VALIDATION COMPLETED!")

    train_data.unpersist()
    test_data.unpersist()
    df.unpersist()


if __name__ == "__main__":

    try:
        main()

    finally:
        spark.stop()
