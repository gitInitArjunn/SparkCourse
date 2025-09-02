import logging
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)
print(">>> Active log level for Spark:", logging.getLogger("pyspark").getEffectiveLevel())
