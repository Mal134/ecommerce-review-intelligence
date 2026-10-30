
import os
import getpass
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window

import psycopg2
from psycopg2.extras import execute_batch


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["SPARK_LOCAL_HOSTNAME"] = "localhost"

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data/processed/reviews.parquet"

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5433,
    "database": "ecommerce_sentiment_db",
    "user": "postgres"
}

BATCH_SIZE = 500


# --------------------------------------------------
# 2. SPARK SESSION
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("LoadReviewsToPostgreSQL")
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
# 3. DATABASE INSERT QUERY
# --------------------------------------------------

INSERT_QUERY = """
INSERT INTO reviews (
    review_id,
    clothing_id,
    age,
    title,
    review_text,
    rating,
    recommended_ind,
    positive_feedback_count,
    division_name,
    department_name,
    class_name,
    cleaned_text,
    sentiment_label,
    sentiment_name
)
VALUES (
    %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s
)
ON CONFLICT (review_id)
DO UPDATE SET
    clothing_id = EXCLUDED.clothing_id,
    age = EXCLUDED.age,
    title = EXCLUDED.title,
    review_text = EXCLUDED.review_text,
    rating = EXCLUDED.rating,
    recommended_ind = EXCLUDED.recommended_ind,
    positive_feedback_count = EXCLUDED.positive_feedback_count,
    division_name = EXCLUDED.division_name,
    department_name = EXCLUDED.department_name,
    class_name = EXCLUDED.class_name,
    cleaned_text = EXCLUDED.cleaned_text,
    sentiment_label = EXCLUDED.sentiment_label,
    sentiment_name = EXCLUDED.sentiment_name;
"""


# --------------------------------------------------
# 4. MAIN IMPORT FUNCTION
# --------------------------------------------------

def main():

    password = getpass.getpass(
        "Enter your PostgreSQL password: "
    )

    DB_CONFIG["password"] = password

    connection = None

    try:

        print("\nConnecting to PostgreSQL...")

        connection = psycopg2.connect(**DB_CONFIG)

        print("PostgreSQL connection successful!")

        print("\nReading processed Parquet data...")

        df = spark.read.parquet(str(INPUT_PATH))

        columns = [
            "clothing_id",
            "age",
            "title",
            "review_text",
            "rating",
            "recommended_ind",
            "positive_feedback_count",
            "division_name",
            "department_name",
            "class_name",
            "cleaned_text",
            "label",
            "sentiment"
        ]

        df = df.select(*columns)

        total_records = df.count()

        print("Records to import:", total_records)

        # Generate unique IDs for the PostgreSQL primary key.
        # Original dataset and Parquet file remain unchanged.
        ordering_columns = [
            col(c).asc_nulls_first()
            for c in columns
        ]

        window = Window.orderBy(*ordering_columns)

        df = df.withColumn(
            "review_id",
            row_number().over(window)
        )

        df = df.select(
            "review_id",
            "clothing_id",
            "age",
            "title",
            "review_text",
            "rating",
            "recommended_ind",
            "positive_feedback_count",
            "division_name",
            "department_name",
            "class_name",
            "cleaned_text",
            col("label").alias("sentiment_label"),
            col("sentiment").alias("sentiment_name")
        )

        cursor = connection.cursor()

        batch = []
        imported = 0

        print("\nStarting batch import...")

        for row in df.toLocalIterator():

            batch.append(tuple(row))

            if len(batch) >= BATCH_SIZE:

                execute_batch(
                    cursor,
                    INSERT_QUERY,
                    batch,
                    page_size=BATCH_SIZE
                )

                imported += len(batch)
                print(f"Processed {imported}/{total_records}")

                batch.clear()

        # Insert remaining records
        if batch:

            execute_batch(
                cursor,
                INSERT_QUERY,
                batch,
                page_size=BATCH_SIZE
            )

            imported += len(batch)

        connection.commit()

        print("\nImport completed successfully!")

        print("Records processed:", imported)

        cursor.close()

        # Verify database count
        verify_cursor = connection.cursor()

        verify_cursor.execute(
            "SELECT COUNT(*) FROM reviews;"
        )

        database_count = verify_cursor.fetchone()[0]

        print("\nPostgreSQL verification:")
        print("Total reviews in database:", database_count)

        verify_cursor.close()

    except Exception as error:

        if connection:
            connection.rollback()

        print("\nIMPORT FAILED:")
        print(error)

        raise

    finally:

        if connection:
            connection.close()

        spark.stop()


if __name__ == "__main__":
    main()
