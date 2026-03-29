"""
data/loader.py
==============
Dataset loading utilities shared by all four implementation notebooks
and the benchmark notebook.

Dataset: UCI Car Evaluation
  All six feature columns are categorical strings — no discretisation needed.
  This simplifies the pipeline compared to continuous-feature datasets like Iris.

  Columns : buying, maint, doors, persons, lug_boot, safety → class (label)
  Classes : unacc, acc, good, vgood
  Size    : 1728 rows, balanced across features but skewed on class (~70% unacc)

Two loaders are provided:
  - load_rdd()       → (train_rdd, test_rdd)   each row = (label, [feat_0...feat_5])
  - load_dataframe() → (train_df, test_df)      named columns + "label" column

Both use the same random seed so train/test splits are identical and results
are comparable across all four implementations.
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType


# ---------------------------------------------------------------------------
# Column definitions — update these if you switch to a different dataset
# ---------------------------------------------------------------------------

# TODO: If using a different dataset, change COLUMN_NAMES to match your CSV header.
COLUMN_NAMES = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "label"]

# Feature columns are everything except the label
FEATURE_COLS = COLUMN_NAMES[:-1]

LABEL_COL = "label"

# Fixed seed ensures train/test split is identical across all four notebooks,
# so timing comparisons are fair (same data, not just same distribution).
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Hardcoded 5-row dummy dataset for immediate local testing
# Exactly the same format as the real car.data file: 7 comma-separated values.
# ---------------------------------------------------------------------------
DUMMY_DATA = [
    ("low", "med", "2",    "2", "small", "low",  "unacc"),
    ("low", "med", "2",    "2", "small", "med",  "unacc"),
    ("low", "med", "2",    "2", "small", "high", "acc"),
    ("low", "med", "2",    "2", "med",   "low",  "acc"),
    ("low", "med", "2",    "2", "med",   "med",  "good"),
]


# ---------------------------------------------------------------------------
# SparkSession — local mode for development/testing
# ---------------------------------------------------------------------------
def get_spark():
    """
    Create or retrieve a SparkSession.
    On Databricks, getOrCreate() returns the already-running cluster session.
    Locally, it starts an embedded Spark instance using all available CPU cores.
    """
    return (
        SparkSession.builder
        .appName("NaiveBayes-Loader")
        .master("local[*]")
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# RDD loader
# ---------------------------------------------------------------------------
def load_rdd(spark, filepath=None, train_ratio=0.8):
    """
    Load the car evaluation dataset and return (train_rdd, test_rdd) as Spark RDDs.

    Each RDD element is a tuple: (label, [feature_0, feature_1, ..., feature_5])
    where all values are strings (no numeric conversion needed for this dataset).

    Args:
        spark       : active SparkSession
        filepath    : path to the CSV/data file; pass None to use DUMMY_DATA
        train_ratio : fraction used for training (default 0.8 = 80/20 split)

    Returns:
        (train_rdd, test_rdd)
    """
    sc = spark.sparkContext

    if filepath is None:
        # Use the hardcoded dummy dataset so the pipeline can be tested immediately.
        # Each row is already a Python tuple matching the expected column order.
        raw_rdd = sc.parallelize(DUMMY_DATA)

    else:
        # TODO: Change filepath to your actual file location before running.
        #   Local   : "/Users/you/data/car.data"
        #   Databricks DBFS: "dbfs:/FileStore/car.data"
        raw_rdd = sc.textFile(filepath)

        # TODO: The real car.data file has no header row and uses "," as delimiter.
        #       If your file has a header, add:
        #           header = raw_rdd.first()
        #           raw_rdd = raw_rdd.filter(lambda line: line != header)
        #       If your delimiter is different (e.g. ";"), change split(",") below.
        raw_rdd = raw_rdd.map(lambda line: tuple(line.strip().split(",")))

    # Reformat each row from a flat tuple to (label, [features]).
    # The label is the last column; features are all preceding columns.
    # Input:  ("low", "med", "2", "2", "small", "low", "unacc")
    # Output: ("unacc", ["low", "med", "2", "2", "small", "low"])
    parsed_rdd = raw_rdd.map(lambda row: (row[-1], list(row[:-1])))

    # Split into train and test with a fixed seed so all notebooks use the same split.
    # TODO: Adjust train_ratio if you want a different split (e.g. 0.7 for 70/30).
    train_rdd, test_rdd = parsed_rdd.randomSplit(
        [train_ratio, 1.0 - train_ratio], seed=RANDOM_SEED
    )

    # Cache training data — it will be read multiple times during the training phase.
    train_rdd.cache()

    return train_rdd, test_rdd


# ---------------------------------------------------------------------------
# DataFrame loader
# ---------------------------------------------------------------------------
def load_dataframe(spark, filepath=None, train_ratio=0.8):
    """
    Load the car evaluation dataset and return (train_df, test_df) as Spark DataFrames.

    The returned DataFrames have columns:
        buying, maint, doors, persons, lug_boot, safety, label
    All columns are string type (the dataset is entirely categorical).

    Args:
        spark       : active SparkSession
        filepath    : path to the data file; pass None to use DUMMY_DATA
        train_ratio : fraction used for training

    Returns:
        (train_df, test_df)
    """
    schema = StructType([
        StructField(col, StringType(), True) for col in COLUMN_NAMES
    ])

    if filepath is None:
        # Build a DataFrame directly from the in-memory dummy data.
        df = spark.createDataFrame(DUMMY_DATA, schema=schema)

    else:
        # TODO: Change filepath to your actual file location.
        #   Databricks DBFS: "dbfs:/FileStore/car.data"
        # TODO: The real car.data file has no header. If yours does, change
        #       header="false" → header="true" and remove the schema argument.
        # TODO: Change delimiter if your file doesn't use commas.
        df = (
            spark.read
            .option("header", "false")
            .option("delimiter", ",")
            .schema(schema)
            .csv(filepath)
        )

    # Rename the last column to "label" for consistency with LABEL_COL constant.
    # (The schema already names it "label", so this is a no-op unless you changed it.)
    # TODO: If your dataset has a different label column name, add a rename here:
    #       df = df.withColumnRenamed("class", "label")

    # Split with the same seed used in load_rdd() so both loaders produce
    # equivalent splits and results are directly comparable.
    # TODO: Adjust train_ratio to match what you use in load_rdd().
    train_df, test_df = df.randomSplit(
        [train_ratio, 1.0 - train_ratio], seed=RANDOM_SEED
    )

    train_df.cache()

    return train_df, test_df


# ---------------------------------------------------------------------------
# Quick sanity check — run this file directly to verify both loaders work
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    spark = get_spark()

    print("=== RDD Loader (dummy data) ===")
    train_rdd, test_rdd = load_rdd(spark)
    print(f"  Train rows : {train_rdd.count()}")
    print(f"  Test rows  : {test_rdd.count()}")
    print(f"  Sample     : {train_rdd.first()}")

    print("\n=== DataFrame Loader (dummy data) ===")
    train_df, test_df = load_dataframe(spark)
    print(f"  Train rows : {train_df.count()}")
    print(f"  Test rows  : {test_df.count()}")
    train_df.show(3)

    spark.stop()
