from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, IntegerType, StructField, FloatType

schema = StructType([
    StructField("userID", IntegerType(), True),
    StructField("itemID", IntegerType(), True),
    StructField("value", FloatType(), True)
])

spark = SparkSession.builder.appName("Total-order-value-DF").getOrCreate()

df = spark.read.csv("Data/customer-orders.csv", header = True, schema = schema)

df.groupBy("userID")\
    .agg(F.round(F.sum("value"), 2).alias("total_order_value"))\
    .orderBy(F.desc("total_order_value"))\
    .show(50)

spark.stop()
