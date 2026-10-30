
import os
import json
from pathlib import Path

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

import numpy as np
import xgboost as xgb

from scipy.sparse import csr_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
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
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PROCESSED_DIR = BASE_DIR / "data/processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("TunedXGBoostSentiment")
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

def create_feature_pipeline():

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
# 5. CLASS WEIGHTS
# --------------------------------------------------

def calculate_weights(y, minority_multiplier):

    classes, counts = np.unique(y, return_counts=True)

    total = len(y)
    number_of_classes = len(classes)

    weights = {
        int(class_id): total / (number_of_classes * int(count))
        for class_id, count in zip(classes, counts)
    }

    # Increase Negative and Neutral weights slightly.
    weights[0] *= minority_multiplier
    weights[1] *= minority_multiplier

    sample_weights = np.array([
        weights[int(label)] for label in y
    ])

    return sample_weights, weights


# --------------------------------------------------
# 6. XGBOOST CONFIGURATIONS
# --------------------------------------------------

CONFIGURATIONS = [
    {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 180,
        "min_child_weight": 2,
        "minority_multiplier": 1.0
    },
    {
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 180,
        "min_child_weight": 2,
        "minority_multiplier": 1.0
    },
    {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 180,
        "min_child_weight": 2,
        "minority_multiplier": 1.15
    },
    {
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 220,
        "min_child_weight": 3,
        "minority_multiplier": 1.15
    }
]


def create_model(config):

    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        min_child_weight=config["min_child_weight"],
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=2
    )


# --------------------------------------------------
# 7. MAIN
# --------------------------------------------------

def main():

    print("\n" + "=" * 60)
    print("TUNED XGBOOST SENTIMENT ANALYSIS")
    print("=" * 60)

    # Load processed data
    df = spark.read.parquet(str(INPUT_PATH))

    df = (
        df.select("cleaned_text", "label")
        .filter(
            col("cleaned_text").isNotNull()
            & (col("cleaned_text") != "")
            & col("label").isin(0, 1, 2)
        )
        .withColumn("label", col("label").cast("double"))
    )

    df = df.cache()

    print("\nTotal records:", df.count())

    # Same original train-test split
    train_data, test_data = df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    train_data = train_data.cache()
    test_data = test_data.cache()

    print("Training records:", train_data.count())
    print("Testing records:", test_data.count())

    # Split training data into fitting and validation portions
    fit_data, validation_data = train_data.randomSplit(
        [0.8, 0.2],
        seed=123
    )

    fit_data = fit_data.cache()
    validation_data = validation_data.cache()

    print("Model fitting records:", fit_data.count())
    print("Validation records:", validation_data.count())

    # --------------------------------------------------
    # 8. FIT FEATURE PIPELINE ON FITTING DATA ONLY
    # --------------------------------------------------

    print("\nFitting validation feature pipeline...")

    validation_pipeline = create_feature_pipeline()
    validation_feature_model = validation_pipeline.fit(fit_data)

    fit_features = validation_feature_model.transform(fit_data)
    validation_features = validation_feature_model.transform(
        validation_data
    )

    X_fit, y_fit = convert_features(fit_features)
    X_validation, y_validation = convert_features(
        validation_features
    )

    print("Fitting matrix:", X_fit.shape)
    print("Validation matrix:", X_validation.shape)

    # --------------------------------------------------
    # 9. TEST CONFIGURATIONS
    # --------------------------------------------------

    tuning_results = []
    best_macro_f1 = -1
    best_config = None

    for index, config in enumerate(CONFIGURATIONS, start=1):

        print("\n" + "-" * 55)
        print(f"CONFIGURATION {index}/{len(CONFIGURATIONS)}")
        print(config)

        sample_weights, _ = calculate_weights(
            y_fit,
            config["minority_multiplier"]
        )

        model = create_model(config)

        model.fit(
            X_fit,
            y_fit,
            sample_weight=sample_weights,
            verbose=False
        )

        validation_predictions = model.predict(
            X_validation
        ).astype(int)

        macro_f1 = f1_score(
            y_validation,
            validation_predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        )

        accuracy = accuracy_score(
            y_validation,
            validation_predictions
        )

        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Macro-F1: {macro_f1:.4f}")

        tuning_results.append({
            "configuration": config,
            "validation_accuracy": float(accuracy),
            "validation_macro_f1": float(macro_f1)
        })

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_config = config.copy()

    print("\n" + "=" * 60)
    print("BEST CONFIGURATION")
    print(best_config)
    print(f"Best validation Macro-F1: {best_macro_f1:.4f}")

    # --------------------------------------------------
    # 10. REFIT FEATURES ON FULL TRAINING DATA
    # --------------------------------------------------

    print("\nRefitting TF-IDF on complete training data...")

    final_pipeline = create_feature_pipeline()
    final_feature_model = final_pipeline.fit(train_data)

    final_train_features = final_feature_model.transform(train_data)
    final_test_features = final_feature_model.transform(test_data)

    X_train, y_train = convert_features(final_train_features)
    X_test, y_test = convert_features(final_test_features)

    print("Final training matrix:", X_train.shape)
    print("Final testing matrix:", X_test.shape)

    # --------------------------------------------------
    # 11. TRAIN FINAL TUNED MODEL
    # --------------------------------------------------

    print("\nTraining final tuned XGBoost model...")

    final_weights, class_weights = calculate_weights(
        y_train,
        best_config["minority_multiplier"]
    )

    final_model = create_model(best_config)

    final_model.fit(
        X_train,
        y_train,
        sample_weight=final_weights,
        verbose=False
    )

    predictions = final_model.predict(X_test).astype(int)

    # --------------------------------------------------
    # 12. FINAL EVALUATION
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, predictions)

    weighted_precision = precision_score(
        y_test, predictions,
        average="weighted", zero_division=0
    )

    weighted_recall = recall_score(
        y_test, predictions,
        average="weighted", zero_division=0
    )

    weighted_f1 = f1_score(
        y_test, predictions,
        average="weighted", zero_division=0
    )

    macro_f1 = f1_score(
        y_test, predictions,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    class_report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=["Negative", "Neutral", "Positive"],
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2]
    )

    print("\n" + "=" * 60)
    print("FINAL TUNED XGBOOST RESULTS")
    print("=" * 60)

    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall:    {weighted_recall:.4f}")
    print(f"Weighted F1:        {weighted_f1:.4f}")
    print(f"Macro-F1:           {macro_f1:.4f}")

    print("\nCLASS-WISE REPORT")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=["Negative", "Neutral", "Positive"],
        zero_division=0
    ))

    print("\nCONFUSION MATRIX")
    print("Rows = Actual | Columns = Predicted")
    print(matrix)

    # --------------------------------------------------
    # 13. SAVE MODEL AND PREDICTIONS
    # --------------------------------------------------

    model_path = MODEL_DIR / "xgboost_tuned_model.json"
    final_model.save_model(str(model_path))

    prediction_path = (
        PROCESSED_DIR / "xgboost_tuned_predictions.parquet"
    )

    prediction_rows = [
        (
            int(actual),
            int(predicted),
            ["Negative", "Neutral", "Positive"][int(predicted)]
        )
        for actual, predicted in zip(y_test, predictions)
    ]

    prediction_df = spark.createDataFrame(
        prediction_rows,
        ["actual_label", "prediction", "predicted_sentiment"]
    )

    prediction_df.write.mode("overwrite").parquet(
        str(prediction_path)
    )

    # --------------------------------------------------
    # 14. SAVE EVALUATION REPORT
    # --------------------------------------------------

    report = {
        "model": "Tuned XGBoost",
        "training_records": len(y_train),
        "testing_records": len(y_test),
        "best_configuration": best_config,
        "validation_best_macro_f1": float(best_macro_f1),
        "class_weights": {
            str(key): float(value)
            for key, value in class_weights.items()
        },
        "accuracy": float(accuracy),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "class_wise_metrics": {
            name: {
                "precision": float(class_report[name]["precision"]),
                "recall": float(class_report[name]["recall"]),
                "f1_score": float(class_report[name]["f1-score"])
            }
            for name in ["Negative", "Neutral", "Positive"]
        },
        "confusion_matrix": matrix.tolist(),
        "tuning_results": tuning_results
    }

    report_path = RESULTS_DIR / "xgboost_tuned_evaluation.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nSaved model:", model_path)
    print("Saved predictions:", prediction_path)
    print("Saved report:", report_path)

    # Cleanup
    fit_data.unpersist()
    validation_data.unpersist()
    train_data.unpersist()
    test_data.unpersist()
    df.unpersist()

    print("\nTUNED XGBOOST COMPLETED!")


if __name__ == "__main__":

    try:
        main()

    finally:
        spark.stop()
