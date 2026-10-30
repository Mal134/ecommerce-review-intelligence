
import os
import json
from pathlib import Path

# --------------------------------------------------
# 1. SPARK NETWORK CONFIGURATION
# --------------------------------------------------

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import NaiveBayes, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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
# 3. CREATE SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("ECommerceSentimentAnalysis")
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

def evaluate_model(predictions, model_name):

    print("\n" + "=" * 60)
    print(f"{model_name} EVALUATION")
    print("=" * 60)

    # Keep only valid prediction and label values
    prediction_data = (
        predictions
        .select(
            col("prediction").cast("double"),
            col("label").cast("double")
        )
        .dropna()
    )

    # Convert to RDD for class-wise metrics
    prediction_and_labels = (
        prediction_data
        .rdd
        .map(lambda row: (row["prediction"], row["label"]))
    )

    metrics = MulticlassMetrics(prediction_and_labels)

    # Overall metrics
    accuracy = float(metrics.accuracy)
    weighted_precision = float(metrics.weightedPrecision)
    weighted_recall = float(metrics.weightedRecall)
    weighted_f1 = float(metrics.weightedFMeasure())

    print("\nOVERALL METRICS")
    print("-" * 40)
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Weighted Recall:    {weighted_recall:.4f}")
    print(f"Weighted F1-score:  {weighted_f1:.4f}")

    # Class-wise metrics
    class_names = {
        0.0: "Negative",
        1.0: "Neutral",
        2.0: "Positive"
    }

    class_results = {}
    class_f1_scores = []

    print("\nCLASS-WISE METRICS")
    print("-" * 40)

    for class_id, class_name in class_names.items():

        precision = float(metrics.precision(class_id))
        recall = float(metrics.recall(class_id))
        f1 = float(metrics.fMeasure(class_id))

        class_results[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        class_f1_scores.append(f1)

        print(f"\n{class_name}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")

    # Macro F1 = unweighted average of class F1 scores
    macro_f1 = sum(class_f1_scores) / len(class_f1_scores)

    print("\nMacro-F1:", f"{macro_f1:.4f}")

    # Confusion matrix
    confusion_matrix = metrics.confusionMatrix().toArray()

    print("\nCONFUSION MATRIX")
    print("Rows = Actual | Columns = Predicted")
    print("Class order: Negative, Neutral, Positive")
    print(confusion_matrix)

    # Save metrics in dictionary
    results = {
        "model": model_name,
        "accuracy": accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "class_wise_metrics": class_results,
        "confusion_matrix": confusion_matrix.tolist()
    }

    return results


# --------------------------------------------------
# 5. MAIN TRAINING FUNCTION
# --------------------------------------------------

def main():

    print("\n" + "=" * 60)
    print("E-COMMERCE SENTIMENT ANALYSIS")
    print("=" * 60)

    # Check input file
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {INPUT_PATH}"
        )

    # Load processed Parquet dataset
    print("\nLoading processed dataset...")

    df = spark.read.parquet(str(INPUT_PATH))

    # Select required columns
    df = (
        df.select("cleaned_text", "label", "sentiment")
        .filter(
            col("cleaned_text").isNotNull()
            & (col("cleaned_text") != "")
            & col("label").isin(0, 1, 2)
        )
        .withColumn("label", col("label").cast("double"))
    )

    df = df.cache()

    total_records = df.count()

    print(f"Total records: {total_records}")

    print("\nSentiment distribution:")
    df.groupBy("sentiment").count().orderBy("sentiment").show()

    # Verify all three classes are present
    label_counts = {
        int(row["label"]): row["count"]
        for row in df.groupBy("label").count().collect()
    }

    print("Label counts:", label_counts)

    if set(label_counts.keys()) != {0, 1, 2}:
        raise ValueError(
            "Dataset must contain all three sentiment classes."
        )

    # --------------------------------------------------
    # 6. TRAIN-TEST SPLIT
    # --------------------------------------------------

    print("\nSplitting dataset into training and testing sets...")

    train_data, test_data = df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    train_data = train_data.cache()
    test_data = test_data.cache()

    train_count = train_data.count()
    test_count = test_data.count()

    print(f"Training records: {train_count}")
    print(f"Testing records:  {test_count}")

    print("\nTraining class distribution:")
    train_data.groupBy("label").count().orderBy("label").show()

    print("\nTesting class distribution:")
    test_data.groupBy("label").count().orderBy("label").show()

    # --------------------------------------------------
    # 7. TEXT PROCESSING PIPELINE
    # --------------------------------------------------

    print("\nCreating TF-IDF text processing pipeline...")

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
    # 8. DEFINE MODELS
    # --------------------------------------------------

    models = {
        "Naive Bayes": NaiveBayes(
            featuresCol="features",
            labelCol="label",
            smoothing=1.0,
            modelType="multinomial"
        ),

        "Logistic Regression": LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=50,
            regParam=0.01,
            elasticNetParam=0.0
        )
    }

    all_results = []

    # --------------------------------------------------
    # 9. TRAIN AND EVALUATE MODELS
    # --------------------------------------------------

    for model_name, classifier in models.items():

        print("\n" + "#" * 60)
        print(f"TRAINING: {model_name}")
        print("#" * 60)

        pipeline = Pipeline(
            stages=[
                tokenizer,
                remover,
                vectorizer,
                idf,
                classifier
            ]
        )

        # Train model
        print(f"\nFitting {model_name}...")

        fitted_model = pipeline.fit(train_data)

        # Generate test predictions
        print("Generating predictions...")

        predictions = fitted_model.transform(test_data).cache()

        predictions.count()

        # Evaluate
        results = evaluate_model(
            predictions,
            model_name
        )

        all_results.append(results)

        # Create safe filename
        filename = model_name.lower().replace(" ", "_")

        # Save model
        model_path = MODEL_DIR / f"{filename}_model"

        fitted_model.write().overwrite().save(str(model_path))

        print(f"\nModel saved: {model_path}")

        # Save predictions
        prediction_path = (
            PROCESSED_DIR / f"{filename}_predictions.parquet"
        )

        predictions.write.mode("overwrite").parquet(
            str(prediction_path)
        )

        print(f"Predictions saved: {prediction_path}")

        predictions.unpersist()

    # --------------------------------------------------
    # 10. MODEL COMPARISON
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    for result in all_results:

        print(f"\n{result['model']}")
        print(f"Accuracy:   {result['accuracy']:.4f}")
        print(f"Weighted F1:{result['weighted_f1']:.4f}")
        print(f"Macro-F1:   {result['macro_f1']:.4f}")

    # Select best model using Macro-F1
    best_model = max(
        all_results,
        key=lambda x: x["macro_f1"]
    )

    print("\nBEST MODEL BASED ON MACRO-F1:")
    print(best_model["model"])
    print(f"Macro-F1: {best_model['macro_f1']:.4f}")

    # --------------------------------------------------
    # 11. SAVE EVALUATION REPORT
    # --------------------------------------------------

    report_path = RESULTS_DIR / "model_evaluation.json"

    report = {
        "dataset_records": total_records,
        "training_records": train_count,
        "testing_records": test_count,
        "sentiment_labels": {
            "0": "Negative",
            "1": "Neutral",
            "2": "Positive"
        },
        "models": all_results,
        "best_model_by_macro_f1": best_model["model"]
    }

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print(f"\nEvaluation report saved: {report_path}")

    # --------------------------------------------------
    # 12. CLEANUP
    # --------------------------------------------------

    train_data.unpersist()
    test_data.unpersist()
    df.unpersist()

    print("\n" + "=" * 60)
    print("MODEL TRAINING AND EVALUATION COMPLETED!")
    print("=" * 60)


# --------------------------------------------------
# 13. EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    try:
        main()

    except Exception as error:
        print("\nERROR OCCURRED:")
        print(error)
        raise

    finally:
        spark.stop()

