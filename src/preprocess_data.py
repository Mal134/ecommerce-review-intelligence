
import os
import re
from pathlib import Path

# ============================================================
# 1. SPARK CONFIGURATION
# ============================================================

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_DIR / "data" / "raw" /
    "Womens Clothing E-Commerce Reviews.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR / "data" / "processed" /
    "reviews.parquet"
)

# ============================================================
# 2. CREATE SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("EcommerceSentimentPreprocessing")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .config("spark.hadoop.fs.default.name", "file:///")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:

    print("\n========== SPARK CONFIGURATION ==========")
    print("Spark version:", spark.version)
    print("Project directory:", PROJECT_DIR)

    # ========================================================
    # 3. CHECK INPUT FILE
    # ========================================================

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at:\n{INPUT_PATH}"
        )

    print("\n========== DATASET LOADING ==========")
    print("Input file:", INPUT_PATH)

    # ========================================================
    # 4. LOAD CSV
    # ========================================================

    raw_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")
        .option("mode", "PERMISSIVE")
        .option("encoding", "UTF-8")
        .option("multiLine", "true")
        .csv(str(INPUT_PATH))
    )

    original_count = raw_df.count()

    print("Original records:", original_count)
    print("Original columns:", raw_df.columns)

    # ========================================================
    # 5. NORMALIZE COLUMN NAMES
    # ========================================================

    def normalize_column_name(name):
        name = name.strip().lower()
        name = re.sub(r"[^a-z0-9]+", "_", name)
        return name.strip("_")

    normalized_columns = [
        normalize_column_name(column)
        for column in raw_df.columns
    ]

    df = raw_df.toDF(*normalized_columns)

    if "unnamed_0" in df.columns:
        df = df.drop("unnamed_0")

    # ========================================================
    # 6. VALIDATE REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "clothing_id",
        "age",
        "title",
        "review_text",
        "rating",
        "recommended_ind",
        "positive_feedback_count",
        "division_name",
        "department_name",
        "class_name"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}\n"
            f"Available columns: {df.columns}"
        )

    # ========================================================
    # 7. CLEAN EMPTY TEXT VALUES
    # ========================================================

    text_columns = [
        "title",
        "review_text",
        "division_name",
        "department_name",
        "class_name"
    ]

    for column in text_columns:
        df = df.withColumn(
            column,
            F.when(
                F.trim(F.col(column)) == "",
                None
            ).otherwise(F.trim(F.col(column)))
        )

    # ========================================================
    # 8. SAFE NUMERIC CONVERSION
    # ========================================================

    df = (
        df
        .withColumn(
            "age",
            F.expr("try_cast(`age` AS INT)")
        )
        .withColumn(
            "rating",
            F.expr("try_cast(`rating` AS INT)")
        )
        .withColumn(
            "recommended_ind",
            F.expr("try_cast(`recommended_ind` AS INT)")
        )
        .withColumn(
            "positive_feedback_count",
            F.expr("try_cast(`positive_feedback_count` AS INT)")
        )
    )

    # Keep Clothing ID as a string identifier.
    # Remove surrounding whitespace.
    df = df.withColumn(
        "clothing_id",
        F.trim(F.col("clothing_id"))
    )

    # ========================================================
    # 9. HANDLE MISSING CATEGORY VALUES
    # ========================================================

    category_columns = [
        "division_name",
        "department_name",
        "class_name"
    ]

    for column in category_columns:
        df = df.withColumn(
            column,
            F.coalesce(F.col(column), F.lit("Unknown"))
        )

    # ========================================================
    # 10. COMBINE TITLE AND REVIEW TEXT
    # ========================================================

    df = df.withColumn(
        "combined_text",
        F.concat_ws(
            " ",
            F.coalesce(F.col("title"), F.lit("")),
            F.coalesce(F.col("review_text"), F.lit(""))
        )
    )

    df = df.withColumn(
        "combined_text",
        F.trim(F.col("combined_text"))
    )

    # ========================================================
    # 11. TEXT CLEANING
    # ========================================================

    df = df.withColumn(
        "cleaned_text",
        F.lower(F.col("combined_text"))
    )

    df = df.withColumn(
        "cleaned_text",
        F.regexp_replace(
            F.col("cleaned_text"),
            r"[^\p{L}\p{N}\s]",
            " "
        )
    )

    df = df.withColumn(
        "cleaned_text",
        F.trim(
            F.regexp_replace(
                F.col("cleaned_text"),
                r"\s+",
                " "
            )
        )
    )

    # ========================================================
    # 12. REMOVE INVALID RECORDS
    # ========================================================

    # A valid record must have:
    # 1. Non-empty review text.
    # 2. A rating between 1 and 5.
    # 3. A numeric Clothing ID.

    text_valid_df = df.filter(
        F.col("cleaned_text").isNotNull() &
        (F.length(F.col("cleaned_text")) > 0) &
        F.col("rating").isNotNull() &
        F.col("rating").between(1, 5)
    )

    # Count malformed Clothing IDs before filtering.
    invalid_clothing_id_count = (
        text_valid_df
        .filter(
            F.col("clothing_id").isNull() |
            (~F.col("clothing_id").rlike("^[0-9]+$"))
        )
        .count()
    )

    # Retain only numeric Clothing IDs.
    valid_df = text_valid_df.filter(
        F.col("clothing_id").isNotNull() &
        F.col("clothing_id").rlike("^[0-9]+$")
    )

    processed_count = valid_df.count()
    excluded_count = original_count - processed_count

    print("\n========== CLOTHING ID VALIDATION ==========")
    print("Records with invalid Clothing IDs:",
          invalid_clothing_id_count)
    print("Records retained after validation:",
          processed_count)

    # ========================================================
    # 13. CREATE SENTIMENT LABELS
    # ========================================================

    # Rating-based weak supervision:
    # 1, 2 -> Negative -> 0
    # 3    -> Neutral  -> 1
    # 4, 5 -> Positive -> 2

    valid_df = (
        valid_df
        .withColumn(
            "sentiment",
            F.when(F.col("rating") <= 2, "Negative")
             .when(F.col("rating") == 3, "Neutral")
             .when(F.col("rating") >= 4, "Positive")
        )
        .withColumn(
            "label",
            F.when(F.col("rating") <= 2, 0)
             .when(F.col("rating") == 3, 1)
             .when(F.col("rating") >= 4, 2)
             .cast("int")
        )
    )

    # ========================================================
    # 14. SELECT FINAL COLUMNS
    # ========================================================

    final_columns = [
        "clothing_id",
        "age",
        "title",
        "review_text",
        "combined_text",
        "cleaned_text",
        "rating",
        "sentiment",
        "label",
        "recommended_ind",
        "positive_feedback_count",
        "division_name",
        "department_name",
        "class_name"
    ]

    processed_df = valid_df.select(*final_columns)

    # ========================================================
    # 15. PREPROCESSING RESULTS
    # ========================================================

    print("\n========== PREPROCESSING RESULTS ==========")
    print("Original records:", original_count)
    print("Processed records:", processed_count)
    print("Records excluded:", excluded_count)

    print("\nFinal dataset shape:")
    print(f"{processed_count} rows x {len(final_columns)} columns")

    # ========================================================
    # 16. SENTIMENT DISTRIBUTION
    # ========================================================

    print("\n========== SENTIMENT DISTRIBUTION ==========")

    processed_df.groupBy("sentiment") \
        .count() \
        .orderBy("sentiment") \
        .show(truncate=False)

    # ========================================================
    # 17. RATING DISTRIBUTION
    # ========================================================

    print("\n========== RATING DISTRIBUTION ==========")

    processed_df.groupBy("rating") \
        .count() \
        .orderBy("rating") \
        .show()

    # ========================================================
    # 18. SAMPLE PROCESSED RECORDS
    # ========================================================

    print("\n========== SAMPLE PROCESSED RECORDS ==========")

    processed_df.select(
        "clothing_id",
        "rating",
        "sentiment",
        "label",
        "cleaned_text"
    ).show(10, truncate=60)

    # ========================================================
    # 19. VERIFY IDs BEFORE SAVING
    # ========================================================

    remaining_invalid_ids = processed_df.filter(
        F.col("clothing_id").isNull() |
        (~F.col("clothing_id").rlike("^[0-9]+$"))
    ).count()

    if remaining_invalid_ids != 0:
        raise ValueError(
            f"Found {remaining_invalid_ids} invalid Clothing IDs. "
            "Parquet will not be saved."
        )

    print("\nAll Clothing IDs passed validation.")

    # ========================================================
    # 20. SAVE PROCESSED DATA AS PARQUET
    # ========================================================

    print("\n========== SAVING PROCESSED DATA ==========")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        processed_df
        .write
        .mode("overwrite")
        .parquet(str(OUTPUT_PATH))
    )

    print("Parquet file saved successfully!")
    print("Output location:", OUTPUT_PATH)

    # ========================================================
    # 21. VERIFY SAVED PARQUET
    # ========================================================

    print("\n========== PARQUET VERIFICATION ==========")

    verified_df = spark.read.parquet(str(OUTPUT_PATH))

    verified_count = verified_df.count()

    verified_invalid_ids = verified_df.filter(
        F.col("clothing_id").isNull() |
        (~F.col("clothing_id").rlike("^[0-9]+$"))
    ).count()

    print("Records saved:", verified_count)
    print("Expected records:", processed_count)
    print("Invalid IDs in saved Parquet:", verified_invalid_ids)

    if (
        verified_count == processed_count
        and verified_invalid_ids == 0
    ):
        print("\nPARQUET VERIFICATION SUCCESSFUL!")
    else:
        raise ValueError(
            "Parquet verification failed. "
            "Check the saved dataset before proceeding."
        )

    # Show saved IDs and their corresponding reviews.
    print("\n========== SAVED DATA SAMPLE ==========")

    verified_df.select(
        "clothing_id",
        "review_text"
    ).show(10, truncate=80)

    print("\n========== FINAL DATA SCHEMA ==========")
    verified_df.printSchema()

finally:
    spark.stop()
    print("\n========== PREPROCESSING COMPLETED ==========")


