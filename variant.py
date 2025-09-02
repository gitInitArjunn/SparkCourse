from pyspark.sql import SparkSession
from pyspark.sql.functions import parse_json, variant_get, try_variant_get, col

spark = SparkSession.builder.appName("VariantExample").getOrCreate()

data = [
    (1, '{"name":"Alice","age":30,"city":"New York"}'),
    (2, '{"product":"Laptop","price":1200.99,"specs":{"RAM":"16GB","CPU":"Intel i7"}}'),
    (3, '{"event":"login","user":"bob123","timestamp":"2025-02-14T10:00:00Z"}')
]

# Create a DF, parse the JSON string straight into a VARIANT column
df = (
    spark
      .createDataFrame(data, ["id", "json_str"])
      .select(
        col("id").cast("int"),
        parse_json("json_str").alias("data")
      )
)

df.printSchema()
# root
#  |-- id: integer (nullable = true)
#  |-- data: variant (nullable = true)

df.show(truncate=False)

# Extract a few fields out of the VARIANT
df.select(
    col("id"),
    variant_get(col("data"), "$.name", "string").alias("name"),
    try_variant_get(col("data"), "$.age", "int").alias("age"),
    try_variant_get(col("data"), "$.specs.RAM", "string").alias("ram")
).show(truncate=False)
