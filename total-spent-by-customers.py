import os
from pyspark import SparkConf, SparkContext

Data = os.getenv("Data", ".")

def parse(line):
    """Parse given line to retrieve customer id and value of customer order"""
    fields = line.split(',')
    customer_id = int(fields[0])
    value = float(fields[1])  # Convert to float
    return (customer_id, value)

conf = SparkConf().setMaster("local").setAppName("TotalSpentByCustomer")
sc = SparkContext(conf=conf)

lines = sc.textFile(f"{Data}/customer-orders.csv")
splitData = lines.map(parse)
totalByCustomer = splitData.reduceByKey(lambda x, y: x + y).collect()

totalByCustomer = sorted(totalByCustomer, key = lambda x: x[1], reverse = True)

for identity, total in totalByCustomer:
    print(f"{identity}\t{total:.2f}")
