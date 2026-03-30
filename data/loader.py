"""
data/loader.py
==============
Dataset loading utilities shared by all four implementation notebooks
and the benchmark notebook.

Two datasets are supported:
  - UCI Car Evaluation  (1 728 rows, 6 features, 4 classes, all categorical)
  - UCI Mushroom        (8 124 rows, 22 features, 2 classes, all categorical)

Both datasets are entirely categorical, so no discretisation step is needed
before running the Naive Bayes algorithm.

Two loader pairs are provided per dataset:
  load_car_rdd()           → (train_rdd, test_rdd)
  load_car_dataframe()     → (train_df,  test_df)
  load_mushroom_rdd()      → (train_rdd, test_rdd)
  load_mushroom_dataframe()→ (train_df,  test_df)

Each RDD element  : (label, [feat_0, feat_1, ..., feat_n])
Each DataFrame row: named feature columns + a "label" column

Both loaders for the same dataset use the same random seed so train/test
splits are identical and RDD vs DataFrame results are directly comparable.
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType


# ---------------------------------------------------------------------------
# SparkSession — local mode for development/testing
# ---------------------------------------------------------------------------
def get_spark():
    """
    Create or retrieve a SparkSession.

    On Databricks, a SparkSession is already running and getOrCreate() returns it.
    The .master() call is intentionally omitted when running on a cluster — Databricks
    injects the master URL automatically. Calling .master("local[*]") on a cluster
    would override the cluster config and run everything on the driver node only.

    Locally (laptop / CI), we fall back to local[*] so the code runs without a cluster.
    We detect Databricks by checking for the DATABRICKS_RUNTIME_VERSION env variable,
    which Databricks sets automatically on every cluster node.
    """
    import os
    builder = SparkSession.builder.appName("NaiveBayes-Loader")
    if not os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        # Running locally — start an embedded Spark instance.
        builder = builder.master("local[*]")
    # On Databricks, no .master() call — the cluster provides its own URL.
    return builder.getOrCreate()


# ===========================================================================
# CAR EVALUATION DATASET
# ===========================================================================

# Column definitions — update these if you switch to a different dataset.
CAR_COLUMN_NAMES = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]
CAR_FEATURE_COLS = CAR_COLUMN_NAMES[:-1]   # all columns except the label
CAR_LABEL_COL    = "label"

# In car.data the label is the LAST column.
_CAR_LABEL_INDEX = -1

# Backward-compat aliases — notebooks import these by the shorter names.
# Both resolve to the car evaluation dataset constants.
FEATURE_COLS = CAR_FEATURE_COLS
LABEL_COL    = CAR_LABEL_COL

# Fixed seed ensures train/test split is identical across all four notebooks,
# so timing comparisons are fair (same data, not just same distribution).
RANDOM_SEED = 42

# Hardcoded 5-row dummy dataset for immediate local testing.
# Same format as the real car.data: comma-separated, label last.
CAR_DUMMY_DATA = [
    ("low", "med", "2", "2", "small", "low",  "unacc"),
    ("low", "med", "2", "2", "small", "med",  "unacc"),
    ("low", "med", "2", "2", "small", "high", "acc"),
    ("low", "med", "2", "2", "med",   "low",  "acc"),
    ("low", "med", "2", "2", "med",   "med",  "good"),
]


def load_car_rdd(spark, filepath=None, train_ratio=0.8):
    """
    Load the UCI Car Evaluation dataset and return (train_rdd, test_rdd).

    Each RDD element: (label, [feat_0, ..., feat_5])
    All values are strings — no numeric conversion needed.

    Args:
        spark       : active SparkSession
        filepath    : path to car.data; pass None to use CAR_DUMMY_DATA
                      Local:      "/path/to/car.data"
                      Databricks: "dbfs:/FileStore/car.data"
        train_ratio : fraction used for training (default 0.8)

    Returns:
        (train_rdd, test_rdd)
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
    Load the UCI Car Evaluation dataset and return (train_df, test_df).

    Columns: buying, maint, doors, persons, lug_boot, safety, label
    All columns are StringType.

    Args:
        spark       : active SparkSession
        filepath    : path to car.data; pass None to use CAR_DUMMY_DATA
        train_ratio : fraction used for training

    Returns:
        (train_df, test_df)
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


# ===========================================================================
# MUSHROOM DATASET
# ===========================================================================

# In mushroom.data the label is the FIRST column (p=poisonous, e=edible),
# followed by 22 categorical feature columns.
# Source: https://archive.ics.uci.edu/ml/datasets/Mushroom
MUSHROOM_COLUMN_NAMES = [
    "label",                       # p = poisonous, e = edible  ← FIRST column
    "cap_shape",                   # b,c,x,f,k,s
    "cap_surface",                 # f,g,y,s
    "cap_color",                   # n,b,c,g,r,p,u,e,w,y
    "bruises",                     # t,f
    "odor",                        # a,l,c,y,f,m,n,p,s
    "gill_attachment",             # a,d,f,n
    "gill_spacing",                # c,w,d
    "gill_size",                   # b,n
    "gill_color",                  # k,n,b,h,g,r,o,p,u,e,w,y
    "stalk_shape",                 # e,t
    "stalk_root",                  # b,c,u,e,z,r,?
    "stalk_surface_above_ring",    # f,y,k,s
    "stalk_surface_below_ring",    # f,y,k,s
    "stalk_color_above_ring",      # n,b,c,g,o,p,e,w,y
    "stalk_color_below_ring",      # n,b,c,g,o,p,e,w,y
    "veil_type",                   # p,u
    "veil_color",                  # n,o,w,y
    "ring_number",                 # n,o,t
    "ring_type",                   # c,e,f,l,n,p,s,z
    "spore_print_color",           # k,n,b,h,r,o,u,w,y
    "population",                  # a,c,n,s,v,y
    "habitat",                     # g,l,m,p,u,w,d
]

MUSHROOM_FEATURE_COLS = MUSHROOM_COLUMN_NAMES[1:]  # everything after the label
MUSHROOM_LABEL_COL    = "label"

# Label is the FIRST column in mushroom.data (opposite of car.data).
_MUSHROOM_LABEL_INDEX = 0

# Hardcoded dummy rows for immediate local smoke-testing.
# Format matches the real mushroom.data: label first, then 22 feature values.
MUSHROOM_DUMMY_DATA = [
    ("p", "x", "s", "n", "t", "p", "f", "c", "n", "k", "e", "e", "s", "s", "w", "w", "p", "w", "o", "p", "k", "s", "u"),
    ("e", "x", "s", "y", "t", "a", "f", "c", "b", "k", "e", "c", "s", "s", "w", "w", "p", "w", "o", "p", "n", "n", "g"),
    ("e", "b", "s", "w", "t", "l", "f", "c", "b", "n", "e", "c", "s", "s", "w", "w", "p", "w", "o", "p", "n", "n", "m"),
    ("p", "x", "y", "w", "t", "p", "f", "c", "n", "n", "e", "e", "s", "s", "w", "w", "p", "w", "o", "p", "k", "s", "u"),
    ("e", "x", "s", "g", "f", "n", "f", "w", "b", "k", "t", "e", "s", "s", "w", "w", "p", "w", "o", "e", "n", "a", "g"),
]


def load_mushroom_rdd(spark, filepath=None, train_ratio=0.8):
    """
    Load the UCI Mushroom dataset and return (train_rdd, test_rdd).

    Each RDD element: (label, [feat_0, ..., feat_21])
    label is "p" (poisonous) or "e" (edible).
    All values are single-character strings — no numeric conversion needed.

    Note on the raw file format:
        mushroom.data has NO header row.
        The label is the FIRST column (unlike car.data where it is last).
        Delimiter is comma.

    Args:
        spark       : active SparkSession
        filepath    : path to mushroom.data; pass None to use MUSHROOM_DUMMY_DATA
                      Local:      "/path/to/mushroom.data"
                      Databricks: "dbfs:/FileStore/mushroom.data"
        train_ratio : fraction used for training (default 0.8)

    Returns:
        (train_rdd, test_rdd)
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

    # Same seed as car loaders so cross-dataset comparisons are consistent.
    train_rdd, test_rdd = parsed_rdd.randomSplit(
        [train_ratio, 1.0 - train_ratio], seed=RANDOM_SEED
    )
    train_rdd.cache()
    return train_rdd, test_rdd


def load_mushroom_dataframe(spark, filepath=None, train_ratio=0.8):
    """
    Load the UCI Mushroom dataset and return (train_df, test_df).

    Columns: label, cap_shape, cap_surface, ..., habitat  (23 total)
    All columns are StringType.

    Args:
        spark       : active SparkSession
        filepath    : path to mushroom.data; pass None to use MUSHROOM_DUMMY_DATA
        train_ratio : fraction used for training

    Returns:
        (train_df, test_df)
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


# ===========================================================================
# Quick sanity check — run this file directly to verify all loaders work
# ===========================================================================
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
