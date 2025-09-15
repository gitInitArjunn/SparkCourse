"""Spark SQL Example script"""
import os
from pyspark.sql import SparkSession
from pyspark.sql import Row

# Create a SparkSession
spark = SparkSession.builder.appName("SparkSQL").getOrCreate()

def mapper(line):
    """"Map dataset to rows of a df"""
    fields = line.split(',')
    return Row(ID=int(fields[0]), name=str(fields[1].encode("utf-8")), \
              age=int(fields[2]), numFriends=int(fields[3]))

Data = os.getenv("Data", ".")
lines = spark.sparkContext.textFile(f"{Data}/fakefriends.csv")
people = lines.map(mapper)

# Infer the schema, and register the DataFrame as a table.
schemaPeople = spark.createDataFrame(people).cache()
schemaPeople.createOrReplaceTempView("people")

# SQL can be run over DataFrames that have been registered as a table.
teenagers = spark.sql("SELECT * FROM people WHERE age >= 13 AND age <= 19")

# The results of SQL queries are RDDs and support all the normal RDD operations.
# for teen in teenagers.collect():
#     print(teen)
teenagers.show(5)

# We can also use functions instead of SQL queries:
schemaPeople.groupBy("age").count().orderBy("age").show()

spark.stop()
