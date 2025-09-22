from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType

# ---------------------
# Cosine Similarity
# ---------------------
def computeCosineSimilarity(data):
    pairScores = data.withColumn("xx", F.col("rating1") * F.col("rating1")) \
                     .withColumn("yy", F.col("rating2") * F.col("rating2")) \
                     .withColumn("xy", F.col("rating1") * F.col("rating2"))

    similarity = pairScores.groupBy("movie1", "movie2") \
        .agg(
            F.sum("xy").alias("numerator"),
            (F.sqrt(F.sum("xx")) * F.sqrt(F.sum("yy"))).alias("denominator"),
            F.count("xy").alias("numPairs")
        ) \
        .withColumn("score",
            F.when(F.col("denominator") != 0, F.col("numerator") / F.col("denominator"))
             .otherwise(0)
        ) \
        .select("movie1", "movie2", "score", "numPairs")

    return similarity


# ---------------------
# Main Script
# ---------------------
spark = SparkSession.builder.appName("MovieSimilarities").master("local[*]").getOrCreate()

# Schemas
movieNamesSchema = StructType([
    StructField("movieID", IntegerType(), True),
    StructField("movieTitle", StringType(), True)
])

moviesSchema = StructType([
    StructField("userID", IntegerType(), True),
    StructField("movieID", IntegerType(), True),
    StructField("rating", IntegerType(), True),
    StructField("timestamp", LongType(), True)
])

# Load datasets
movieNames = spark.read.option("sep", "|").option("charset", "ISO-8859-1") \
    .schema(movieNamesSchema).csv("Data/ml-100k/u.item")

movies = spark.read.option("sep", "\t").schema(moviesSchema).csv("Data/ml-100k/u.data")

ratings = movies.select("userID", "movieID", "rating")

# Build movie pairs
moviePairs = ratings.alias("r1") \
    .join(ratings.alias("r2"),
          (F.col("r1.userID") == F.col("r2.userID")) &
          (F.col("r1.movieID") < F.col("r2.movieID"))) \
    .select(
        F.col("r1.movieID").alias("movie1"),
        F.col("r2.movieID").alias("movie2"),
        F.col("r1.rating").alias("rating1"),
        F.col("r2.rating").alias("rating2")
    )

# Compute similarities
moviePairSimilarities = computeCosineSimilarity(moviePairs).cache()

# ---------------------
# Example Query
# ---------------------
def getSimilarMovies(movieID, scoreThreshold=0.97, coOccurrenceThreshold=50, topN=10):
    filtered = moviePairSimilarities.filter(
        ((F.col("movie1") == movieID) | (F.col("movie2") == movieID)) &
        (F.col("score") > scoreThreshold) &
        (F.col("numPairs") > coOccurrenceThreshold)
    )
    results = filtered.orderBy(F.col("score").desc()).limit(topN)

    # Bring movie names in via join (instead of collect lookups)
    names = movieNames.select("movieID", "movieTitle")
    resultsWithNames = results \
        .join(names, results.movie1 == names.movieID, "left") \
        .withColumnRenamed("movieTitle", "movie1Title") \
        .drop("movieID") \
        .join(names, results.movie2 == names.movieID, "left") \
        .withColumnRenamed("movieTitle", "movie2Title") \
        .drop("movieID")

    return resultsWithNames


# Example usage
movieID = 50  # just hardcode or pass dynamically
similarMovies = getSimilarMovies(movieID)
similarMovies.show(truncate=False)
