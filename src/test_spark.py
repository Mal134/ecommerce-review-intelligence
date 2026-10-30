
import os

# Force Spark to use a resolvable local network address
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("EcommerceSentimentTest")
    .master("local[2]")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

data = [
    (1, "Excellent product", 5),
    (2, "Very poor quality", 1),
    (3, "It is average", 3)
]

df = spark.createDataFrame(
    data,
    ["review_id", "review_text", "rating"]
)

print("\n========== SPARK VALIDATION ==========")
print("Spark version:", spark.version)
print("Master:", spark.sparkContext.master)
print("Total records:", df.count())

df.show(truncate=False)

print("SPARK TEST SUCCESSFUL")

spark.stop()

