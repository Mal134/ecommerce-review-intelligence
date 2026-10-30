
import os
import json
from pathlib import Path

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

import numpy as np
import xgboost as xgb

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF
)
from pyspark.mllib.evaluation import MulticlassMetrics
from scipy.sparse import csr_matrix


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "reviews.parquet"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("XGBoostSentimentAnalysis")
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
# 3. EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(y_true, y_pred):

    prediction_rdd = spark.sparkContext.parallelize(
        [
            (float(pred), float(actual))
            for actual, pred in zip(y_true, y_pred)
        ]
    )

    metrics = MulticlassMetrics(prediction_rdd)

    accuracy = float(metrics.accuracy)
    weighted_precision = float(metrics.weightedPrecision)
    weighted_recall = float(metrics.weightedRecall)
    weighted_f1 = float(metrics.weightedFMeasure())

    class_names = {
        0.0: "Negative",
        1.0: "Neutral",
        2.0: "Positive"
    }

    class_results = {}
    f1_scores = []

    print("\nOVERALL METRICS")
    print("-" * 40)
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall:    {weighted_recall:.4f}")
    print(f"Weighted F1:        {weighted_f1:.4f}")

    print("\nCLASS-WISE METRICS")
    print("-" * 40)

    for class_id, class_name in class_names.items():

        precision = float(metrics.precision(class_id))
        recall = float(metrics.recall(class_id))
        f1 = float(metrics.fMeasure(class_id))

        f1_scores.append(f1)

        class_results[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        print(f"\n{class_name}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")

    macro_f1 = sum(f1_scores) / len(f1_scores)

    confusion_matrix = metrics.confusionMatrix().toArray()

    print(f"\nMacro-F1: {macro_f1:.4f}")
    print("\nCONFUSION MATRIX")
    print("Rows = Actual | Columns = Predicted")
    print("Order: Negative, Neutral, Positive")
    print(confusion_matrix)

    return {
        "model": "XGBoost",
        "accuracy": accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "class_wise_metrics": class_results,
        "confusion_matrix": confusion_matrix.tolist()
    }


# --------------------------------------------------
# 4. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\n" + "=" * 60)
    print("XGBOOST SENTIMENT ANALYSIS")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    # Load data
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

    # --------------------------------------------------
    # 5. TRAIN-TEST SPLIT
    # --------------------------------------------------

    train_data, test_data = df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    train_data = train_data.cache()
    test_data = test_data.cache()

    print("Training records:", train_data.count())
    print("Testing records:", test_data.count())

    # --------------------------------------------------
    # 6. SPARK TF-IDF PIPELINE
    # --------------------------------------------------

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

    feature_pipeline = Pipeline(
        stages=[tokenizer, remover, vectorizer, idf]
    )

    print("\nFitting TF-IDF feature pipeline...")

    feature_model = feature_pipeline.fit(train_data)

    train_features = feature_model.transform(train_data)
    test_features = feature_model.transform(test_data)

    # --------------------------------------------------
    # 7. CONVERT SPARK FEATURES TO SCIPY SPARSE MATRIX
    # --------------------------------------------------

    print("\nConverting features to sparse matrices...")

    def convert_features(data):

        rows = data.select("features", "label").collect()

        labels = np.array(
            [int(row["label"]) for row in rows],
            dtype=np.int32
        )

        matrix = csr_matrix(
            np.vstack([
                row["features"].toArray()
                for row in rows
            ]),
            dtype=np.float32
        )

        return matrix, labels

    X_train, y_train = convert_features(train_features)
    X_test, y_test = convert_features(test_features)

    print("Training matrix:", X_train.shape)
    print("Testing matrix:", X_test.shape)

    # --------------------------------------------------
    # 8. CLASS WEIGHTS
    # --------------------------------------------------

    unique_classes, class_counts = np.unique(
        y_train,
        return_counts=True
    )

    total = len(y_train)
    class_weights = {
        int(class_id): total / (
            len(unique_classes) * int(count)
        )
        for class_id, count in zip(unique_classes, class_counts)
    }

    sample_weights = np.array([
        class_weights[int(label)]
        for label in y_train
    ])

    print("\nClass weights:", class_weights)

    # --------------------------------------------------
    # 9. TRAIN XGBOOST
    # --------------------------------------------------

    model = xgb.XGBClassifier(
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

    print("\nTraining XGBoost...")

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        verbose=False
    )

    print("Training completed!")

    # --------------------------------------------------
    # 10. PREDICTIONS
    # --------------------------------------------------

    print("\nGenerating predictions...")

    predictions = model.predict(X_test).astype(int)

    results = evaluate_model(y_test, predictions)

    # --------------------------------------------------
    # 11. SAVE MODEL
    # --------------------------------------------------

    model_path = MODEL_DIR / "xgboost_model.json"
    model.save_model(str(model_path))

    print("\nModel saved:", model_path)

    # --------------------------------------------------
    # 12. SAVE PREDICTIONS
    # --------------------------------------------------

    prediction_path = (
        PROCESSED_DIR / "xgboost_predictions.parquet"
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

    print("Predictions saved:", prediction_path)

    # --------------------------------------------------
    # 13. SAVE REPORT
    # --------------------------------------------------

    report = {
        "dataset_records": df.count(),
        "training_records": len(y_train),
        "testing_records": len(y_test),
        "class_weights": {
            str(key): float(value)
            for key, value in class_weights.items()
        },
        "model_results": results
    }

    report_path = RESULTS_DIR / "xgboost_evaluation.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nEvaluation report saved:", report_path)

    # Cleanup
    train_data.unpersist()
    test_data.unpersist()
    df.unpersist()

    print("\n" + "=" * 60)
    print("XGBOOST TRAINING COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":

    try:
        main()

    finally:
        spark.stop()
