from pyspark.sql import functions
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("FriendsByAgeDF").getOrCreate()

df = spark.read.csv("Data/fakefriends-header.csv", header = True, inferSchema = True)

df_filtered = df.groupBy("age").count().orderBy("age")

df_filtered.show()

spark.stop()
