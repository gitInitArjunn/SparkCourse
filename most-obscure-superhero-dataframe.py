from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

spark = SparkSession.builder.appName("MostPopularSuperhero").getOrCreate()

schema = StructType([ \
                     StructField("id", IntegerType(), True), \
                     StructField("name", StringType(), True)])

names = spark.read.schema(schema).option("sep", " ").csv("Data/Marvel-names")

lines = spark.read.text("Data/Marvel-graph")

# Small tweak vs. what's shown in the video: we trim each line of whitespace as that could
# throw off the counts.
connections = lines.withColumn("id", func.split(func.trim(func.col("value")), " ")[0]) \
    .withColumn("connections", func.size(func.split(func.trim(func.col("value")), " ")) - 1) \
    .groupBy("id").agg(func.sum("connections").alias("connections"))

bottom10 = connections.filter(func.col("connections") > 1) \
                        .sort(func.col("connections").asc()) \
                        .limit(10)

bottom10WithNames = bottom10.join(names, bottom10.id == names.id, "inner") \
                        .select("name", "connections") \
                        .orderBy(func.col("connections").asc())

bottom10WithNames.show(10, truncate=False)
