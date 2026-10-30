
import os
from pathlib import Path
from pyspark.sql import SparkSession

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data/processed/reviews.parquet"

spark = (
    SparkSession.builder
    .appName("InspectProcessedDataset")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .config("spark.hadoop.fs.default.name", "file:///")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(str(INPUT_PATH))

print("\nDATASET SCHEMA")
df.printSchema()

print("\nCOLUMN NAMES")
print(df.columns)

print("\nTOTAL RECORDS")
print(df.count())

print("\nSAMPLE RECORD")
df.show(2, truncate=False)

spark.stop()
