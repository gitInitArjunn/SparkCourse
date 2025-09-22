from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, LongType
from pyspark.ml.recommendation import ALS

def load_movie_names(path="Data/ml-100k/u.ITEM"):
    import codecs
    movieNames = {}
    with codecs.open(path, "r", encoding='ISO-8859-1', errors='ignore') as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

def get_user_recommendations(model, userID, names, num_recs=10):
    from pyspark.sql.types import StructType, StructField, IntegerType
    user_df = spark.createDataFrame([[userID]], StructType([StructField("userID", IntegerType(), True)]))
    recs = model.recommendForUserSubset(user_df, num_recs).collect()
    
    print(f"Top {num_recs} recommendations for user ID {userID}:")
    for userRecs in recs:
        for rec in userRecs.recommendations:
            movie = names[rec.movieID]
            print(f"{movie}: {rec.rating:.2f}")

spark = SparkSession.builder.appName("ALSExample").getOrCreate()

moviesSchema = StructType([
    StructField("userID", IntegerType(), True),
    StructField("movieID", IntegerType(), True),
    StructField("rating", IntegerType(), True),
    StructField("timestamp", LongType(), True)
])

names = load_movie_names()
ratings = spark.read.option("sep", "\t").schema(moviesSchema).csv("Data/ml-100k/u.data")

print("Training recommendation model...")
als = ALS(maxIter=5, regParam=0.01, rank=10,
          userCol="userID", itemCol="movieID", ratingCol="rating",
          coldStartStrategy="drop")

model = als.fit(ratings)

# Example usage for user 50
get_user_recommendations(model, 50, names, 10)
