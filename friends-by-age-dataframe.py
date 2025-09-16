from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark session
spark = SparkSession.builder.appName("FriendsByAge").getOrCreate()

# Load dataset
df = spark.read.csv("Data/fakefriends-header.csv", header = True, inferSchema = True)


# Select only relevant columns
friendsByAge = df.select("age", "friends")

# Calculate average number of friends by age
avg_friends_by_age = (
    friendsByAge.groupBy("age")
    .agg(F.round(F.avg("friends"), 2).alias("avg_friends"))
    .orderBy("age")
)

# Show results
avg_friends_by_age.show()

# Stop Spark session
spark.stop()
