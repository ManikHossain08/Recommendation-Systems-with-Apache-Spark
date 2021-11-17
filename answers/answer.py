import os
import sys
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col
from pyspark.sql.functions import desc
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

'''
INTRODUCTION

With this assignment you will get a practical hands-on of recommender
systems in Spark. To begin, make sure you understand the example
at http://spark.apache.org/docs/latest/ml-collaborative-filtering.html
and that you can run it successfully. 

We will use the MovieLens dataset sample provided with Spark and
available in directory `data`.

'''

'''
HELPER FUNCTIONS

These functions are here to help you. Instructions will tell you when
you should use them. Don't modify them!
'''


# Initialize a spark session.
def init_spark():
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    return spark


# Useful functions to print RDDs and Dataframes.
def toCSVLineRDD(rdd):
    '''
    This function convert an RDD or a DataFrame into a CSV string
    '''
    a = rdd.map(lambda row: ",".join([str(elt) for elt in row])) \
        .reduce(lambda x, y: os.linesep.join([x, y]))
    return a + os.linesep


def toCSVLine(data):
    '''
    Convert an RDD or a DataFrame into a CSV string
    '''
    if isinstance(data, RDD):
        return toCSVLineRDD(data)
    elif isinstance(data, DataFrame):
        return toCSVLineRDD(data.rdd)
    return None


def basic_als_recommender(filename, seed):
    '''
    This function must print the RMSE of recommendations obtained
    through ALS collaborative filtering, similarly to the example at
    http://spark.apache.org/docs/latest/ml-collaborative-filtering.html
    The training ratio must be 80% and the test ratio must be 20%. The
    random seed used to sample the training and test sets (passed to
    ''DataFrame.randomSplit') is an argument of the function. The seed
    must also be used to initialize the ALS optimizer (use
    *ALS.setSeed()*). The following parameters must be used in the ALS
    optimizer:
    - maxIter: 5
    - rank: 70
    - regParam: 0.01
    - coldStartStrategy: 'drop'
    Test file: tests/test_basic_als.py
    '''
    spark = init_spark()
    lines = spark.read.text(filename).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), rating=float(p[2])))
    ratings = spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2], seed=seed)
    als = ALS(maxIter=5, rank=70, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop", seed=seed)
    predictions = als.fit(training).transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)

    return rmse


def global_average(filename, seed):
    '''
    This function must print the global average rating for all users and
    all movies in the training set. Training and test
    sets should be determined as before (e.g: as in function basic_als_recommender).
    Test file: tests/test_global_average.py
    '''
    spark = init_spark()
    lines = spark.read.text(filename).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2])))
    ratings = spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2], seed=seed)

    result = training.agg({"rating": "avg"}).collect()[0]
    return result["avg(rating)"]


def global_average_recommender(filename, seed):
    '''
    This function must print the RMSE of recommendations obtained
    through global average, that is, the predicted rating for each
    user-movie pair must be the global average computed in the previous
    task. Training and test
    sets should be determined as before. You can add a column to an existing DataFrame with function *.withColumn(...)*.
    Test file: tests/test_global_average_recommender.py
    '''
    spark = init_spark()
    lines = spark.read.text(filename).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2])))
    ratings = spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2], seed=seed)
    average = training.agg({"rating": "avg"}).collect()[0]['avg(rating)']
    training = training.withColumn('avg(rating)', lit(average))

    als = ALS(maxIter=5, rank=70, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="avg(rating)",
              coldStartStrategy="drop", seed=seed)
    predictions = als.fit(training).transform(test)

    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)

    return rmse


def means_and_interaction(filename, seed, n):
    '''
    This function must return the *n* first elements of a DataFrame
    containing, for each (userId, movieId, rating) triple, the
    corresponding user mean (computed on the training set), item mean
    (computed on the training set) and user-item interaction *i* defined
    as *i=rating-(user_mean+item_mean-global_mean)*. *n* must be passed on
    the command line. The DataFrame must contain the following columns:

    - userId # as in the input file
    - movieId #  as in the input file
    - rating # as in the input file
    - user_mean # computed on the training set
    - item_mean # computed on the training set
    - user_item_interaction # i = rating - (user_mean+item_mean-global_mean)

    Rows must be ordered by ascending userId and then by ascending movieId.

    Training and test sets should be determined as before.
    Test file: tests/test_means_and_interaction.py

    Note, this function should return a list of collected Rows. Please, have a
    look at the test file to ensure you have the right format.
    '''
    spark = init_spark()
    lines = spark.read.text(filename).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2])))
    ratings = spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2], seed=seed)

    user_mean = training.groupBy("userId").agg({"rating": "avg"})
    item_mean = training.groupBy("movieId").agg({"rating": "avg"})

    training = training.join(item_mean, on=["movieId"], how='left_outer').withColumnRenamed("avg(rating)", "item_mean")
    training = training.join(user_mean, on=["userId"], how='left_outer').withColumnRenamed("avg(rating)", "user_mean")

    global_mean = training.agg({"rating": "avg"}).collect()[0]['avg(rating)']
    training = training.withColumn('user_item_interaction',
                                   col('rating') - (col('user_mean') + col('item_mean') - global_mean))

    return training.orderBy(['userId', 'movieId']).limit(n).collect()


def als_with_bias_recommender(filename, seed):
    '''
    This function must return the RMSE of recommendations obtained
    using ALS + biases. Your ALS model should make predictions for *i*,
    the user-item interaction, then you should recompute the predicted
    rating with the formula *i+user_mean+item_mean-m* (*m* is the
    global rating). The RMSE should compare the original rating column
    and the predicted rating column.  Training and test sets should be
    determined as before. Your ALS model should use the same parameters
    as before and be initialized with the random seed passed as
    parameter. Test file: tests/test_als_with_bias_recommender.py
    '''

    spark = init_spark()
    lines = spark.read.text(filename).rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2])))
    ratings = spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.8, 0.2], seed=seed)

    user_mean = training.groupBy("userId").agg({"rating": "avg"})
    item_mean = training.groupBy("movieId").agg({"rating": "avg"})

    training = training.join(item_mean, on=["movieId"], how='left_outer') \
        .withColumnRenamed("avg(rating)", "item_mean") \
        .join(user_mean, on=["userId"], how='left_outer') \
        .withColumnRenamed("avg(rating)", "user_mean")

    global_mean = training.agg({"rating": "avg"}).collect()[0]['avg(rating)']

    training = training.withColumn('user_item_interaction',
                                   col('rating') - (col('user_mean') + col('item_mean') - global_mean))

    als = ALS(maxIter=5, rank=70, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="user_item_interaction",
              coldStartStrategy="drop", seed=seed)

    intermediate_predicts = als.fit(training).transform(test)

    intermediate_predicts = intermediate_predicts.join(item_mean, on=["movieId"], how='left_outer') \
        .withColumnRenamed("avg(rating)", "item_mean") \
        .join(user_mean, on=["userId"], how='left_outer') \
        .withColumnRenamed("avg(rating)", "user_mean") \
        .withColumnRenamed("prediction", "intermediate_predicts") \
        .withColumn('predicts', col('intermediate_predicts') + col('user_mean') + col('item_mean') - global_mean)

    als = ALS(maxIter=5, rank=70, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="predicts",
              coldStartStrategy="drop", seed=seed)

    predictions = als.fit(intermediate_predicts).transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    return rmse
