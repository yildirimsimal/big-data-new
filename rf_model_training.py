from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

#Spark session
spark = SparkSession.builder.appName("StockPredictionRF").getOrCreate()

#Read cleaned parquet data
df = spark.read.parquet("gs://bero_assignment_bucket/cleaned_data/")

#Label creation (classification target)
df_labeled = df.withColumn("label", 
    when(col("Close") < 0.5, 0)
    .when(col("Close") < 1.5, 1)
    .otherwise(2)
)

#Feature assembly
assembler = VectorAssembler(
    inputCols=["Open", "High", "Low", "Volume"],
    outputCol="features"
)
df_features = assembler.transform(df_labeled).select("features", "label")

#Train-test split
train_data, test_data = df_features.randomSplit([0.8, 0.2], seed=42)

#Random Forest model training
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=10)
model = rf.fit(train_data)

#Predictions
predictions = model.transform(test_data)
predictions.select("prediction", "label", "features").show(5)

#Accuracy evaluation
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print("Test Accuracy:", accuracy)

#Confusion Matrix
prediction_and_labels = predictions.select("prediction", "label") \
    .rdd.map(lambda x: (float(x[0]), float(x[1])))
metrics = MulticlassMetrics(prediction_and_labels)

conf_matrix = metrics.confusionMatrix()
print("Confusion Matrix:")
print(conf_matrix)
