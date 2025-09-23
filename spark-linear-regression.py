from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator

def main():
    # 1. Create Spark session
    spark = SparkSession.builder.appName("LinearRegressionExample").getOrCreate()

    # 2. Load CSV data (assuming multiple features)
    # Example CSV format: feature1,feature2,feature3,label
    df = spark.read.option("header", "false") \
               .option("inferSchema", "true") \
               .csv("Data/regression.txt")

# Rename columns manually if no header
    df = df.toDF("label", "feature1")  # adjust number of features


    # Inspect data
    df.show(5)

    # 3. Combine feature columns into a single features vector
    feature_cols = [col for col in df.columns if col != "label"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df)

    # 4. Scale features
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
    scalerModel = scaler.fit(df)
    df = scalerModel.transform(df).select("label", "scaledFeatures")

    # Rename for ML input
    df = df.withColumnRenamed("scaledFeatures", "features")

    # 5. Split into training and testing datasets
    trainingDF, testDF = df.randomSplit([0.8, 0.2], seed=42)

    # 6. Define Linear Regression model
    lr = LinearRegression(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        maxIter=50,            # increase iterations for convergence
        regParam=0.01,         # regularization strength
        elasticNetParam=0.0,   # 0=L2 (Ridge), 1=L1 (Lasso), 0.5=Elastic Net
        fitIntercept=True
    )

    # 7. Train the model
    model = lr.fit(trainingDF)

    # 8. Make predictions
    predictions = model.transform(testDF)
    predictions.select("label", "prediction").show(20)

    # 9. Evaluate performance
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
    print(f"RMSE on test data: {rmse:.4f}")
    print(f"R² on test data: {r2:.4f}")

    # 10. Inspect model coefficients and intercept
    print("Coefficients: " + str(model.coefficients))
    print("Intercept: " + str(model.intercept))

    spark.stop()

if __name__ == "__main__":
    main()
