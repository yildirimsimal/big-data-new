from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date


spark = SparkSession.builder.appName("StockDataCleaning").getOrCreate()


df_raw = spark.read.csv("gs://bero_assignment_bucket/raw_data/stock_data.csv", header=True, inferSchema=True)


df_clean = df_raw.dropna() \
    .withColumn("Date", to_date(col("Date"), "yyyy-MM-dd")) \
    .withColumn("Open", col("Open").cast("float")) \
    .withColumn("High", col("High").cast("float")) \
    .withColumn("Low", col("Low").cast("float")) \
    .withColumn("Close", col("Close").cast("float")) \
    .withColumn("Adj Close", col("Adj Close").cast("float")) \
    .withColumn("Volume", col("Volume").cast("long"))

#Saving cleaned data as Parquet for optimized performance and compatibility with Spark and Hive. 
df_clean.write.mode("overwrite").parquet("gs://bero_assignment_bucket/cleaned_data/")
