"""Same as word-count.py but makes use of normalization"""
import re
import os
from pyspark import SparkConf, SparkContext

def normalizeWords(text):
    """"Regex to match words that are the same and remove any punctuations"""
    return re.compile(r'\W+', re.UNICODE).split(text.lower())

conf = SparkConf().setMaster("local").setAppName("WordCount")
sc = SparkContext(conf = conf)

Data = os.getenv("Data", ".")
input = sc.textFile("{Data}/book.txt")
words = input.flatMap(normalizeWords)
wordCounts = words.countByValue()

for word, count in wordCounts.items():
    cleanWord = word.encode('ascii', 'ignore')
    if (cleanWord):
        print(cleanWord.decode() + " " + str(count))
