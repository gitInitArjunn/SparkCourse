from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Transform using Spark SQL functions") \
    .config("spark.sql.ansi.enabled", "false") \
    .getOrCreate()

# Create a Spark DataFrame (instead of pandas-on-Spark)
df = spark.createDataFrame([
    (1, "Alice", 25, 50000),
    (2, "Bob", 30, 60000),
    (3, "Charlie", 35, 75000),
    (4, "David", 40, 80000),
    (5, "Emma", 45, 120000)
], ["id", "name", "age", "salary"])

print("Original DataFrame:")
df.show()

# 1. Age in 10 years
df = df.withColumn("age_in_10_years", F.col("age") + 10)

# 2. Salary category (using WHEN..OTHERWISE instead of apply)
df = df.withColumn(
    "salary_category",
    F.when(F.col("salary") < 60000, "Low")
     .when(F.col("salary") < 100000, "Medium")
     .otherwise("High")
)

# 3. Name with age (string concatenation instead of row-wise apply)
df = df.withColumn(
    "name_with_age",
    F.concat_ws(" ", F.col("name"), F.lit("("), F.col("age").cast("string"), F.lit("years old)"))
)

print("\nFinal DataFrame after transformations:")
df.show(truncate=False)

# Stop Spark session
spark.stop()
