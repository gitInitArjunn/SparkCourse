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

    # RMSE (Root Mean Squared Error)
    evaluator_rmse = RegressionEvaluator(
        labelCol="PriceOfUnitArea", 
        predictionCol="prediction", 
        metricName="rmse"
    )
    rmse = evaluator_rmse.evaluate(predictions)

    # R² (coefficient of determination)
    evaluator_r2 = RegressionEvaluator(
        labelCol="PriceOfUnitArea", 
        predictionCol="prediction", 
        metricName="r2"
    )
    r2 = evaluator_r2.evaluate(predictions)

    # MAE (Mean Absolute Error)
    evaluator_mae = RegressionEvaluator(
        labelCol="PriceOfUnitArea", 
        predictionCol="prediction", 
        metricName="mae"
    )
    mae = evaluator_mae.evaluate(predictions)

    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

if __name__ == "__main__":
    main()
