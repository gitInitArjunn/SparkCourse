import os
from pyspark import SparkConf, SparkContext

Data = os.getenv("Data", ".")

def parse(line):
    fields = line.split(',')
    customerID = fields[0]
    value = float(fields[1])  # Convert to float
    return (customerID, value)

conf = SparkConf().setMaster("local").setAppName("TotalSpentByCustomer")
sc = SparkContext(conf=conf)

lines = sc.textFile(f"{Data}/customer-orders.csv")
splitData = lines.map(parse)
totalByCustomer = splitData.reduceByKey(lambda x, y: x + y).collect()

totalByCustomer = sorted(totalByCustomer, key = lambda x: x[1], reverse = True)

for customerID, total in totalByCustomer:
    print(f"{customerID}\t{total}")
