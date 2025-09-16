from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

spark = SparkSession.builder.appName("MinTemperatures").getOrCreate()

# Define schema
schema = StructType([
    StructField("stationID", StringType(), True),
    StructField("date", IntegerType(), True),
    StructField("measure_type", StringType(), True),
    StructField("temperature", FloatType(), True)
])

# Read CSV with schema
df = spark.read.schema(schema).csv("Data/1800.csv")

# Compute min temperature per station and convert to Fahrenheit
minTempsByStationF = (
    df.filter(F.col("measure_type") == "TMIN")
      .groupBy("stationID")
      .agg(F.min("temperature").alias("min_temp_c"))
      .withColumn("temperature_F", F.round(F.col("min_temp_c") * 0.1 * 9/5 + 32, 2))
      .select("stationID", "temperature_F")
      .sort("temperature_F")
)

# Show results
minTempsByStationF.show(truncate=False)

# Optional: collect if you really need driver-side processing
results = minTempsByStationF.collect()
for r in results:
    print(f"{r['stationID']}\t{r['temperature_F']:.2f}F")

spark.stop()
