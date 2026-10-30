
import os
from pathlib import Path

# --------------------------------------------------
# 1. SPARK CONFIGURATION
# --------------------------------------------------

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

PROJECT_DIR = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_DIR / "data" / "processed" / "reviews.parquet"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, length, max as spark_max

# --------------------------------------------------
# 2. CREATE SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("VerifyClothingID")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .config("spark.hadoop.fs.default.name", "file:///")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

try:
    # --------------------------------------------------
    # 3. VERIFY PATH
    # --------------------------------------------------

    print("\n========== PATH VERIFICATION ==========")
    print("Project directory:", PROJECT_DIR)
    print("Script location:", Path(__file__).resolve())
    print("Parquet absolute path:", PARQUET_PATH)
    print("Parquet exists:", PARQUET_PATH.exists())

    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet file not found: {PARQUET_PATH}")

    # --------------------------------------------------
    # 4. READ PARQUET
    # --------------------------------------------------

    df = spark.read.parquet(str(PARQUET_PATH))

    print("\n========== DATASET INFORMATION ==========")
    print("Total records:", df.count())
    print("Columns:", df.columns)

    print("\n========== CLOTHING ID AND REVIEW TEXT ==========")

    df.select(
        "clothing_id",
        "review_text"
    ).show(10, truncate=80)

    # --------------------------------------------------
    # 5. CHECK CLOTHING ID LENGTH
    # --------------------------------------------------

    print("\n========== CLOTHING ID LENGTH ==========")

    df.select(
        spark_max(length(col("clothing_id"))).alias("max_id_length")
    ).show()

    print("\n========== LONGEST CLOTHING IDs ==========")

    df.select(
        "clothing_id",
        length(col("clothing_id")).alias("id_length")
    ).orderBy(
        col("id_length").desc()
    ).show(10, truncate=False)

    # --------------------------------------------------
    # 6. CHECK INVALID VALUES
    # --------------------------------------------------

    invalid_count = df.filter(
        col("clothing_id").isNull() |
        (length(col("clothing_id")) > 50) |
        (col("clothing_id").rlike("[a-zA-Z]"))
    ).count()

    print("\nInvalid Clothing ID records:", invalid_count)

    print("\n========== VERIFICATION COMPLETED ==========")

finally:
    spark.stop()
