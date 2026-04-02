"""
data/loader.py
==============
Dataset loading utilities for the RDD, DataFrame, and benchmark notebooks.
Supports UCI Car Evaluation and UCI Mushroom datasets.
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType


def get_spark():
    # used for local testing only; notebooks create their own SparkSession
    return SparkSession.builder.master("local[*]").appName("NaiveBayes-Loader").getOrCreate()


# CAR EVALUATION DATASET ----------------------------------------------------

# car.csv columns — label is the last column
CAR_COLUMN_NAMES = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]
CAR_FEATURE_COLS = CAR_COLUMN_NAMES[:-1]
CAR_LABEL_COL    = "label"

_CAR_LABEL_INDEX = -1

FEATURE_COLS = CAR_FEATURE_COLS
LABEL_COL    = CAR_LABEL_COL

# fixed seed so all notebooks get the same train/test split
RANDOM_SEED = 42

# small dummy dataset for quick local testing
CAR_DUMMY_DATA = [
    ("low", "med", "2", "2", "small", "low",  "unacc"),
    ("low", "med", "2", "2", "small", "med",  "unacc"),
    ("low", "med", "2", "2", "small", "high", "acc"),
    ("low", "med", "2", "2", "med",   "low",  "acc"),
    ("low", "med", "2", "2", "med",   "med",  "good"),
]


def load_car_rdd(spark, filepath=None, train_ratio=0.8):
    """
    Load the UCI Car Evaluation dataset and return (train_rdd, test_rdd)
    Each RDD element: (label, [feat_0, ..., feat_5])
    """
    sc = spark.sparkContext

    if filepath is None:
        raw_rdd = sc.parallelize(CAR_DUMMY_DATA)
    else:
        raw_rdd = sc.textFile(filepath)
        # car.data has no header row; delimiter is comma.
        raw_rdd = raw_rdd.map(lambda line: tuple(line.strip().split(",")))

    # Input:  ("low", "med", "2", "2", "small", "low", "unacc")
    # Output: ("unacc", ["low", "med", "2", "2", "small", "low"])
    parsed_rdd = raw_rdd.map(lambda row: (row[_CAR_LABEL_INDEX], list(row[:_CAR_LABEL_INDEX])))

    train_rdd, test_rdd = parsed_rdd.randomSplit(
        [train_ratio, 1.0 - train_ratio], seed=RANDOM_SEED
    )
    train_rdd.cache()
    return train_rdd, test_rdd


def load_car_dataframe(spark, filepath=None, train_ratio=0.8):
    """
    Load the UCI Car Evaluation dataset and return (train_df, test_df)
    Columns: buying, maint, doors, persons, lug_boot, safety, label
    """
    schema = StructType([StructField(col, StringType(), True) for col in CAR_COLUMN_NAMES])

    if filepath is None:
        df = spark.createDataFrame(CAR_DUMMY_DATA, schema=schema)
    else:
        df = (
            spark.read
            .option("header", "false")
            .option("delimiter", ",")
            .schema(schema)
            .csv(filepath)
        )

    train_df, test_df = df.randomSplit(
        [train_ratio, 1.0 - train_ratio], seed=RANDOM_SEED
    )
    train_df.cache()
    return train_df, test_df

# MUSHROOM DATASET ----------------------------------------------------------

# label is the FIRST column (p=poisonous, e=edible),
MUSHROOM_COLUMN_NAMES = [
    "label",                       # p = poisonous, e = edible
    "cap_shape",                   
    "cap_surface",                 
    "cap_color",                   
    "bruises",                     
    "odor",                        
    "gill_attachment",             
    "gill_spacing",                
    "gill_size",                   
    "gill_color",                  
    "stalk_shape",                 
    "stalk_root",                  
    "stalk_surface_above_ring",    
    "stalk_surface_below_ring",    
    "stalk_color_above_ring",      
    "stalk_color_below_ring",      
    "veil_type",                   
    "veil_color",                  
    "ring_number",                 
    "ring_type",                   
    "spore_print_color",           
    "population",                  
    "habitat",                     
]

MUSHROOM_FEATURE_COLS = MUSHROOM_COLUMN_NAMES[1:]  # everything after the label
MUSHROOM_LABEL_COL    = "label"

# label is the FIRST column in mushroom.csv
_MUSHROOM_LABEL_INDEX = 0

# dummy rows for local testing
MUSHROOM_DUMMY_DATA = [
    ("p", "x", "s", "n", "t", "p", "f", "c", "n", "k", "e", "e", "s", "s", "w", "w", "p", "w", "o", "p", "k", "s", "u"),
    ("e", "x", "s", "y", "t", "a", "f", "c", "b", "k", "e", "c", "s", "s", "w", "w", "p", "w", "o", "p", "n", "n", "g"),
    ("e", "b", "s", "w", "t", "l", "f", "c", "b", "n", "e", "c", "s", "s", "w", "w", "p", "w", "o", "p", "n", "n", "m"),
    ("p", "x", "y", "w", "t", "p", "f", "c", "n", "n", "e", "e", "s", "s", "w", "w", "p", "w", "o", "p", "k", "s", "u"),
    ("e", "x", "s", "g", "f", "n", "f", "w", "b", "k", "t", "e", "s", "s", "w", "w", "p", "w", "o", "e", "n", "a", "g"),
]


def load_mushroom_rdd(spark, filepath=None, train_ratio=0.8):
    """
    Load the UCI Mushroom dataset and return (train_rdd, test_rdd)
    Each RDD element: (label, [feat_0, ..., feat_21])
    """
    sc = spark.sparkContext

    if filepath is None:
        raw_rdd = sc.parallelize(MUSHROOM_DUMMY_DATA)
    else:
        raw_rdd = sc.textFile(filepath)
        # No header row; label is the first comma-separated value.
        raw_rdd = raw_rdd.map(lambda line: tuple(line.strip().split(",")))

    # Input:  ("p", "x", "s", "n", "t", "p", "f", "c", "n", "k", ...)
    # Output: ("p", ["x", "s", "n", "t", "p", "f", "c", "n", "k", ...])
    parsed_rdd = raw_rdd.map(
        lambda row: (row[_MUSHROOM_LABEL_INDEX], list(row[_MUSHROOM_LABEL_INDEX + 1:]))
    )

    # same seed as car loaders
    train_rdd, test_rdd = parsed_rdd.randomSplit(
        [train_ratio, 1.0 - train_ratio], seed=RANDOM_SEED
    )
    train_rdd.cache()
    return train_rdd, test_rdd


def load_mushroom_dataframe(spark, filepath=None, train_ratio=0.8):
    """
    Load the UCI Mushroom dataset and return (train_df, test_df)
    Columns: label, cap_shape, cap_surface, ..., habitat  (23 total)
    """
    schema = StructType([StructField(col, StringType(), True) for col in MUSHROOM_COLUMN_NAMES])

    if filepath is None:
        df = spark.createDataFrame(MUSHROOM_DUMMY_DATA, schema=schema)
    else:
        df = (
            spark.read
            .option("header", "false")
            .option("delimiter", ",")
            .schema(schema)
            .csv(filepath)
        )

    train_df, test_df = df.randomSplit(
        [train_ratio, 1.0 - train_ratio], seed=RANDOM_SEED
    )
    train_df.cache()
    return train_df, test_df


# Quick sanity check to verify that all loaders work ----------------------

if __name__ == "__main__":
    spark = get_spark()

    print("=== Car RDD Loader (dummy data) ===")
    train_rdd, test_rdd = load_car_rdd(spark)
    print(f"  Train rows : {train_rdd.count()}")
    print(f"  Test rows  : {test_rdd.count()}")
    print(f"  Sample     : {train_rdd.first()}")

    print("\n=== Car DataFrame Loader (dummy data) ===")
    train_df, test_df = load_car_dataframe(spark)
    print(f"  Train rows : {train_df.count()}")
    print(f"  Test rows  : {test_df.count()}")
    train_df.show(3)

    print("\n=== Mushroom RDD Loader (dummy data) ===")
    train_rdd, test_rdd = load_mushroom_rdd(spark)
    print(f"  Train rows : {train_rdd.count()}")
    print(f"  Test rows  : {test_rdd.count()}")
    print(f"  Sample     : {train_rdd.first()}")

    print("\n=== Mushroom DataFrame Loader (dummy data) ===")
    train_df, test_df = load_mushroom_dataframe(spark)
    print(f"  Train rows : {train_df.count()}")
    print(f"  Test rows  : {test_df.count()}")
    train_df.show(3)

    spark.stop()
