
import os
import json
from pathlib import Path

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF
)
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.evaluation import MulticlassMetrics


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "reviews.parquet"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("WeightedSentimentAnalysis")
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
# 3. LOAD DATA
# --------------------------------------------------

def main():

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
    # 4. TRAIN-TEST SPLIT
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
    # 5. CALCULATE CLASS WEIGHTS
    # --------------------------------------------------

    print("\nCalculating class weights...")

    class_counts = {
        int(row["label"]): row["count"]
        for row in train_data.groupBy("label").count().collect()
    }

    number_of_classes = 3
    training_count = sum(class_counts.values())

    weights = {
        class_id: training_count / (
            number_of_classes * class_counts[class_id]
        )
        for class_id in class_counts
    }

    print("Class counts:", class_counts)
    print("Class weights:", weights)


    # --------------------------------------------------
    # 6. APPLY CLASS WEIGHTS
    # --------------------------------------------------

    weighted_data = train_data.withColumn(
        "class_weight",
        when(col("label") == 0.0, lit(weights[0]))
        .when(col("label") == 1.0, lit(weights[1]))
        .when(col("label") == 2.0, lit(weights[2]))
    )

    # --------------------------------------------------
    # 7. TF-IDF PIPELINE
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

    classifier = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="class_weight",
        maxIter=50,
        regParam=0.01,
        elasticNetParam=0.0
    )

    pipeline = Pipeline(
        stages=[
            tokenizer,
            remover,
            vectorizer,
            idf,
            classifier
        ]
    )


    # --------------------------------------------------
    # 8. TRAIN WEIGHTED MODEL
    # --------------------------------------------------

    print("\nTraining class-weighted Logistic Regression...")

    model = pipeline.fit(weighted_data)

    print("Training completed!")


    # --------------------------------------------------
    # 9. GENERATE PREDICTIONS
    # --------------------------------------------------

    print("\nGenerating test predictions...")

    predictions = model.transform(test_data).cache()
    predictions.count()


    # --------------------------------------------------
    # 10. EVALUATION
    # --------------------------------------------------

    prediction_rdd = (
        predictions
        .select("prediction", "label")
        .rdd
        .map(lambda row: (
            float(row["prediction"]),
            float(row["label"])
        ))
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

    print("\n========== WEIGHTED LOGISTIC REGRESSION ==========")

    print("\nOVERALL METRICS")
    print("Accuracy:", round(accuracy, 4))
    print("Weighted Precision:", round(weighted_precision, 4))
    print("Weighted Recall:", round(weighted_recall, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    print("\nCLASS-WISE METRICS")

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
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1-score:", round(f1, 4))

    macro_f1 = sum(f1_scores) / len(f1_scores)

    print("\nMacro-F1:", round(macro_f1, 4))

    confusion_matrix = metrics.confusionMatrix().toArray()

    print("\nCONFUSION MATRIX")
    print("Rows = Actual | Columns = Predicted")
    print(confusion_matrix)


    # --------------------------------------------------
    # 11. SAVE MODEL AND PREDICTIONS
    # --------------------------------------------------

    model_path = MODEL_DIR / "weighted_logistic_regression_model"

    model.write().overwrite().save(str(model_path))

    prediction_path = (
        BASE_DIR / "data" / "processed"
        / "weighted_logistic_regression_predictions.parquet"
    )

    predictions.write.mode("overwrite").parquet(
        str(prediction_path)
    )

    print("\nModel saved:", model_path)
    print("Predictions saved:", prediction_path)


    # --------------------------------------------------
    # 12. SAVE EVALUATION RESULTS
    # --------------------------------------------------

    report = {
        "model": "Weighted Logistic Regression",
        "dataset_records": total_records,
        "training_records": train_data.count(),
        "testing_records": test_data.count(),
        "class_weights": {
            str(key): float(value)
            for key, value in weights.items()
        },
        "accuracy": accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "class_wise_metrics": class_results,
        "confusion_matrix": confusion_matrix.tolist()
    }

    report_path = RESULTS_DIR / "weighted_model_evaluation.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nEvaluation report saved:", report_path)

    predictions.unpersist()
    train_data.unpersist()
    test_data.unpersist()
    df.unpersist()

    print("\nWEIGHTED MODEL TRAINING COMPLETED!")


if __name__ == "__main__":

    try:
        main()

    finally:
        spark.stop()
