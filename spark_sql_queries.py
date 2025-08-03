from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, round, date_format

# Spark session
spark = SparkSession.builder.appName("BigDataProject").getOrCreate()

# Read Parquet data from GCS
df = spark.read.parquet("gs://bero_assignment_bucket/cleaned_data/")

# Create temporary view
df.createOrReplaceTempView("stock_data")

# Query 1: Monthly Average Close
spark.sql("""
SELECT
    date_format(Date, 'yyyy-MM') AS Month,
    ROUND(AVG(Close), 4) AS Avg_Close
FROM stock_data
GROUP BY Month
ORDER BY Month
""").show(10)

# Query 2: Day with Highest Volume
spark.sql("""
SELECT Date, Volume
FROM stock_data
ORDER BY Volume DESC
LIMIT 1
""").show()

# Query 3: Monthly Average Volume
spark.sql("""
SELECT
    date_format(Date, 'yyyy-MM') AS Month,
    ROUND(AVG(Volume), 0) AS Avg_Volume
FROM stock_data
GROUP BY Month
ORDER BY Month
""").show(10)
