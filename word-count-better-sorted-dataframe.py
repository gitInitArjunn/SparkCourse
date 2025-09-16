from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("WordCount").getOrCreate()

# Read input file
inputDF = spark.read.text("Data/book.txt")

# Split -> Filter -> Lowercase -> Count -> Sort (all chained)
wordCounts = (
    inputDF
    .select(F.explode(F.split(F.col("value"), "\\W+")).alias("word"))
    .filter(F.col("word") != "")
    .select(F.lower(F.col("word")).alias("word"))
    .groupBy("word")
    .count()
    .orderBy(F.asc("count"))
)

# Show top 100 words
wordCounts.show(100, truncate=False)

spark.stop()
