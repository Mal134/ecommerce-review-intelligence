
import os
import json
from pathlib import Path

# --------------------------------------------------
# 1. SPARK CONFIGURATION
# --------------------------------------------------

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF
)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.evaluation import MulticlassMetrics


# --------------------------------------------------
# 2. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "reviews.parquet"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 3. SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("RandomForestSentimentAnalysis")
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
# 4. EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(predictions):

    prediction_rdd = (
        predictions
        .select("prediction", "label")
        .rdd
        .map(
            lambda row: (
                float(row["prediction"]),
                float(row["label"])
            )
        )
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

    print(f"\nMacro-F1: {macro_f1:.4f}")

    confusion_matrix = metrics.confusionMatrix().toArray()

    print("\nCONFUSION MATRIX")
    print("Rows = Actual | Columns = Predicted")
    print("Order: Negative, Neutral, Positive")
    print(confusion_matrix)

    return {
        "model": "Random Forest",
        "accuracy": accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "class_wise_metrics": class_results,
        "confusion_matrix": confusion_matrix.tolist()
    }


# --------------------------------------------------
# 5. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("\n" + "=" * 60)
    print("RANDOM FOREST SENTIMENT ANALYSIS")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    # Load processed dataset
    print("\nLoading processed dataset...")

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

    total_records = df.count()

    print("Total records:", total_records)

    # --------------------------------------------------
    # 6. TRAIN-TEST SPLIT
    # --------------------------------------------------

    print("\nSplitting dataset...")

    train_data, test_data = df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    train_data = train_data.cache()
    test_data = test_data.cache()

    train_count = train_data.count()
    test_count = test_data.count()

    print("Training records:", train_count)
    print("Testing records:", test_count)

    # --------------------------------------------------
    # 7. TEXT FEATURE EXTRACTION
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

    # --------------------------------------------------
    # 8. RANDOM FOREST CLASSIFIER
    # --------------------------------------------------

    random_forest = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=50,
        maxDepth=12,
        maxBins=32,
        featureSubsetStrategy="auto",
        seed=42
    )

    pipeline = Pipeline(
        stages=[
            tokenizer,
            remover,
            vectorizer,
            idf,
            random_forest
        ]
    )

    # --------------------------------------------------
    # 9. TRAIN MODEL
    # --------------------------------------------------

    print("\nTraining Random Forest...")
    print("Trees: 50")
    print("Maximum depth: 12")

    model = pipeline.fit(train_data)

    print("\nTraining completed!")

    # --------------------------------------------------
    # 10. GENERATE PREDICTIONS
    # --------------------------------------------------

    print("\nGenerating predictions...")

    predictions = model.transform(test_data).cache()
    predictions.count()

    # --------------------------------------------------
    # 11. EVALUATE MODEL
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("RANDOM FOREST EVALUATION")
    print("=" * 60)

    results = evaluate_model(predictions)

    # --------------------------------------------------
    # 12. SAVE MODEL
    # --------------------------------------------------

    model_path = MODEL_DIR / "random_forest_model"

    model.write().overwrite().save(str(model_path))

    print("\nModel saved:", model_path)

    # --------------------------------------------------
    # 13. SAVE PREDICTIONS
    # --------------------------------------------------

    prediction_path = (
        PROCESSED_DIR / "random_forest_predictions.parquet"
    )

    predictions.write.mode("overwrite").parquet(
        str(prediction_path)
    )

    print("Predictions saved:", prediction_path)

    # --------------------------------------------------
    # 14. SAVE EVALUATION REPORT
    # --------------------------------------------------

    report = {
        "dataset_records": total_records,
        "training_records": train_count,
        "testing_records": test_count,
        "model_results": results
    }

    report_path = RESULTS_DIR / "random_forest_evaluation.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nEvaluation report saved:", report_path)

    # --------------------------------------------------
    # 15. CLEANUP
    # --------------------------------------------------

    predictions.unpersist()
    train_data.unpersist()
    test_data.unpersist()
    df.unpersist()

    print("\n" + "=" * 60)
    print("RANDOM FOREST TRAINING COMPLETED!")
    print("=" * 60)


# --------------------------------------------------
# 16. EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    try:
        main()

    finally:
        spark.stop()
