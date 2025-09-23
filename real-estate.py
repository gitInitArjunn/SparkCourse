from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

def main():
    spark = SparkSession.builder.appName("DecisionTreeRegression").getOrCreate()

    # Load data
    data = spark.read.option("header", "true").option("inferSchema", "true")\
        .csv("Data/realestate.csv")

    # Assemble features
    feature_cols = ["HouseAge", "DistanceToMRT", "NumberConvenienceStores"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(data).select("PriceOfUnitArea", "features")

    # Train-test split
    trainDF, testDF = df.randomSplit([0.7, 0.3], seed=42)

    # Decision Tree Regressor
    dtr = DecisionTreeRegressor(featuresCol="features", labelCol="PriceOfUnitArea", maxDepth=5)
    model = dtr.fit(trainDF)

    # Predictions
    predictions = model.transform(testDF)
    predictions.select("PriceOfUnitArea", "prediction").show(20)

    # Evaluate
    evaluator = RegressionEvaluator(
        labelCol="PriceOfUnitArea",
        predictionCol="prediction",
        metricName="rmse"
    )
    rmse = evaluator.evaluate(predictions)
    r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
    print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # Inspect tree
    print(model.toDebugString)

    spark.stop()

if __name__ == "__main__":
    main()
