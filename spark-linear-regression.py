from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.evaluation import RegressionEvaluator

def main():
    spark = SparkSession.builder.appName("LinearRegression").getOrCreate()

    # Load data as DataFrame
    df = spark.read.option("inferSchema", "true").csv("Data/regression.txt").toDF("label", "feature")

    # Convert feature to Dense Vector
    vectorize = udf(lambda x: Vectors.dense([x]), VectorUDT())
    df = df.withColumn("features", vectorize("feature")).select("label", "features")

    # Train-test split
    trainingDF, testDF = df.randomSplit([0.8, 0.2], seed=42)

    # Linear Regression Model
    lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, fitIntercept=True)
    model = lr.fit(trainingDF)

    # Make predictions
    predictions = model.transform(testDF)
    predictions.select("label", "prediction").show(20)

    # Evaluate
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    print(f"RMSE on test data: {rmse:.4f}")

    spark.stop()

if __name__ == "__main__":
    main()
