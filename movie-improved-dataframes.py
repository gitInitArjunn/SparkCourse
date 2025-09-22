from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ---------------------------
# 1. Spark Session
# ---------------------------
spark = SparkSession.builder.appName("MovieSimilaritiesWithGenres").master("local[*]").getOrCreate()

# ---------------------------
# 2. Schemas
# ---------------------------
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType

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

# ---------------------------
# 3. Load Data
# ---------------------------
movieNames = spark.read.option("sep", "|").option("charset", "ISO-8859-1") \
    .schema(movieNamesSchema).csv("Data/ml-100k/u.item")

movies = spark.read.option("sep", "\t").schema(moviesSchema).csv("Data/ml-100k/u.data")

ratings = movies.select("userID", "movieID", "rating")

# ---------------------------
# 4. Load Genres (Fully DataFrame)
# ---------------------------
rawGenres = spark.read.text("Data/ml-100k/u.item")
# transform genre columns to list of indices where genre=1
movieGenres = rawGenres.withColumn("fields", F.split("value", "\|")) \
    .withColumn("movieID", F.col("fields").getItem(0).cast("int")) \
    .withColumn("genres", F.expr("filter(transform(slice(fields, 6, size(fields)), (x, i) -> IF(x='1', i, null)), x -> x is not null)")) \
    .select("movieID", "genres")

# ---------------------------
# 5. Create Movie Pairs
# ---------------------------
moviePairs = ratings.alias("r1").join(
    ratings.alias("r2"),
    (F.col("r1.userID") == F.col("r2.userID")) & (F.col("r1.movieID") < F.col("r2.movieID"))
).select(
    F.col("r1.movieID").alias("movie1ID"),
    F.col("r2.movieID").alias("movie2ID"),
    F.col("r1.rating").alias("rating1"),
    F.col("r2.rating").alias("rating2")
)

# ---------------------------
# 6. Cosine Similarity Function
# ---------------------------
pairScores = moviePairs.withColumn("xx", F.col("rating1")*F.col("rating1")) \
    .withColumn("yy", F.col("rating2")*F.col("rating2")) \
    .withColumn("xy", F.col("rating1")*F.col("rating2"))

calculateSimilarity = pairScores.groupBy("movie1ID", "movie2ID").agg(
    F.sum("xy").alias("numerator"),
    (F.sqrt(F.sum("xx")) * F.sqrt(F.sum("yy"))).alias("denominator"),
    F.count("xy").alias("numPairs")
)

moviePairSimilarities = calculateSimilarity.withColumn(
    "cosineScore", F.when(F.col("denominator") != 0, F.col("numerator") / F.col("denominator")).otherwise(0)
).select("movie1ID", "movie2ID", "cosineScore", "numPairs")

# ---------------------------
# 7. Genre Overlap
# ---------------------------
genrePairs = movieGenres.alias("g1").join(
    movieGenres.alias("g2"),
    F.col("g1.movieID") < F.col("g2.movieID")
).select(
    F.col("g1.movieID").alias("movie1ID"),
    F.col("g2.movieID").alias("movie2ID"),
    F.col("g1.genres").alias("genres1"),
    F.col("g2.genres").alias("genres2")
)

genreOverlap = genrePairs.withColumn(
    "overlap", F.size(F.array_intersect("genres1", "genres2"))
).withColumn(
    "union", F.size(F.array_union("genres1", "genres2"))
).withColumn(
    "genreScore", F.when(F.col("union") != 0, F.col("overlap")/F.col("union")).otherwise(0)
).select("movie1ID", "movie2ID", "genreScore")

# ---------------------------
# 8. Combine Similarities
# ---------------------------
moviePairFinal = moviePairSimilarities.join(
    genreOverlap, ["movie1ID", "movie2ID"], "inner"
).withColumn(
    "finalScore", F.col("cosineScore") * F.col("genreScore")
)

# ---------------------------
# 9. Add Movie Names
# ---------------------------
moviePairNamed = moviePairFinal.join(
    movieNames.withColumnRenamed("movieID","m1ID").withColumnRenamed("movieTitle","movie1Title"),
    moviePairFinal.movie1ID == F.col("m1ID")
).join(
    movieNames.withColumnRenamed("movieID","m2ID").withColumnRenamed("movieTitle","movie2Title"),
    moviePairFinal.movie2ID == F.col("m2ID")
).select(
    "movie1ID", "movie2ID", "movie1Title", "movie2Title",
    "cosineScore", "genreScore", "finalScore", "numPairs"
)

# ---------------------------
# 10. Top Recommendations for a Given Movie
# ---------------------------
movieID_to_recommend = 50

recommendations = moviePairNamed.filter(
    (F.col("movie1ID") == movieID_to_recommend) | (F.col("movie2ID") == movieID_to_recommend)
).withColumn(
    "similarMovie", F.when(F.col("movie1ID") == movieID_to_recommend, F.col("movie2Title")).otherwise(F.col("movie1Title"))
).select(
    "similarMovie", "cosineScore", "genreScore", "finalScore", "numPairs"
).orderBy(F.col("finalScore").desc())

print(f"Top recommendations for movieID={movieID_to_recommend}:")
recommendations.show(10, truncate=False)
    