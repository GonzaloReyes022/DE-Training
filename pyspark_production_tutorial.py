"""
=============================================================================
PySpark Tutorial for Data/ML Engineers - Production Ready
=============================================================================
Focus: Job interview preparation and production-grade distributed computing.

Why PySpark?
- Process terabytes/petabytes of data
- Distributed computing across clusters
- Native integration with cloud platforms (AWS EMR, Databricks, GCP Dataproc)
- SQL + DataFrame + ML unified API
- Industry standard for big data pipelines

Prerequisites:
    pip install pyspark

Note: This tutorial runs locally. In production, you'd connect to a cluster.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, DoubleType, DateType, TimestampType, ArrayType, MapType
)
import os

# Suppress verbose Spark logs for cleaner output
os.environ['PYSPARK_PYTHON'] = 'python3'

# =============================================================================
# SECTION 1: SparkSession and Basic Operations
# =============================================================================

print("=" * 70)
print("SECTION 1: SparkSession and Basic Operations")
print("=" * 70)

# --- 1.1 Creating SparkSession ---
# SparkSession is the entry point for all Spark functionality

spark = SparkSession.builder \
    .appName("DataEngineerTutorial") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# PRODUCTION CONFIGS (for cluster deployment):
# spark = SparkSession.builder \
#     .appName("ProductionJob") \
#     .config("spark.executor.memory", "8g") \
#     .config("spark.executor.cores", "4") \
#     .config("spark.sql.adaptive.enabled", "true") \
#     .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
#     .enableHiveSupport() \
#     .getOrCreate()

print(f"Spark Version: {spark.version}")
print(f"Spark UI: http://localhost:4040")

# --- 1.2 Creating DataFrames ---

# Method 1: From Python list
data = [
    (1, "Alice", 25, 50000.0, "Engineering"),
    (2, "Bob", 30, 75000.0, "Sales"),
    (3, "Charlie", 35, 60000.0, "Engineering"),
    (4, "Diana", 28, 80000.0, "Marketing"),
    (5, "Eve", 32, 70000.0, "Sales"),
]
columns = ["id", "name", "age", "salary", "department"]

df = spark.createDataFrame(data, columns)
df.show()
df.printSchema()

# Method 2: With explicit schema (PRODUCTION BEST PRACTICE)
schema = StructType([
    StructField("id", IntegerType(), nullable=False),
    StructField("name", StringType(), nullable=False),
    StructField("age", IntegerType(), nullable=True),
    StructField("salary", DoubleType(), nullable=True),
    StructField("department", StringType(), nullable=True)
])

df_with_schema = spark.createDataFrame(data, schema)
print("\nWith explicit schema:")
df_with_schema.printSchema()

# --- 1.3 Basic DataFrame Operations ---

# Select columns
df.select("name", "salary").show()

# Select with expressions
df.select(
    F.col("name"),
    F.col("salary"),
    (F.col("salary") * 1.1).alias("salary_with_raise")
).show()

# Filter rows
df.filter(F.col("age") > 28).show()
df.filter((F.col("department") == "Engineering") & (F.col("salary") > 55000)).show()

# EXERCISE 1.1: Create a DataFrame with the following:
# - Filter employees with salary between 60000 and 80000
# - Select only name, department, and salary
# - Add a column 'tax' = salary * 0.2
# YOUR CODE HERE:
# result = ...


# =============================================================================
# SECTION 2: Data Reading and Writing
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: Data Reading and Writing")
print("=" * 70)

# --- 2.1 Reading Different Formats ---

# CSV (most common)
# df = spark.read.csv("data.csv", header=True, inferSchema=True)

# With explicit options (PRODUCTION PATTERN)
# df = spark.read \
#     .option("header", "true") \
#     .option("inferSchema", "false") \
#     .option("delimiter", ",") \
#     .option("nullValue", "NA") \
#     .option("dateFormat", "yyyy-MM-dd") \
#     .schema(my_schema) \
#     .csv("s3://bucket/path/*.csv")

# Parquet (PREFERRED for production - columnar, compressed)
# df = spark.read.parquet("s3://bucket/data.parquet")

# JSON
# df = spark.read.json("data.json")

# Delta Lake (for ACID transactions)
# df = spark.read.format("delta").load("s3://bucket/delta_table")

# --- 2.2 Writing Data ---

# Write to Parquet (partitioned - PRODUCTION PATTERN)
# df.write \
#     .mode("overwrite") \
#     .partitionBy("department", "year") \
#     .parquet("s3://bucket/output/")

# Write modes:
# - "overwrite": Replace existing data
# - "append": Add to existing data
# - "ignore": Skip if exists
# - "error" (default): Throw error if exists

# --- 2.3 Creating Sample Data for Tutorial ---

# Create sample sales data
sales_data = [
    ("2024-01-15", "Widget", "Electronics", 100, 29.99, "North"),
    ("2024-01-15", "Gadget", "Electronics", 50, 49.99, "South"),
    ("2024-01-16", "Widget", "Electronics", 75, 29.99, "North"),
    ("2024-01-16", "Tool", "Hardware", 200, 15.99, "East"),
    ("2024-01-17", "Gadget", "Electronics", 120, 49.99, "West"),
    ("2024-01-17", "Widget", "Electronics", 90, 29.99, "North"),
    ("2024-01-18", "Tool", "Hardware", 150, 15.99, "South"),
    ("2024-01-18", "Accessory", "Electronics", 300, 9.99, "East"),
    ("2024-01-19", "Widget", "Electronics", 110, 29.99, "West"),
    ("2024-01-19", "Gadget", "Electronics", 80, 49.99, "North"),
]

sales_schema = StructType([
    StructField("date", StringType()),
    StructField("product", StringType()),
    StructField("category", StringType()),
    StructField("quantity", IntegerType()),
    StructField("unit_price", DoubleType()),
    StructField("region", StringType())
])

sales_df = spark.createDataFrame(sales_data, sales_schema)
sales_df = sales_df.withColumn("date", F.to_date("date"))
sales_df.show()


# =============================================================================
# SECTION 3: Transformations - Select, Filter, WithColumn
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Transformations")
print("=" * 70)

# --- 3.1 Column Operations ---

# Add new columns
df_transformed = sales_df \
    .withColumn("revenue", F.col("quantity") * F.col("unit_price")) \
    .withColumn("year", F.year("date")) \
    .withColumn("month", F.month("date")) \
    .withColumn("day_of_week", F.dayofweek("date"))

df_transformed.show()

# Rename columns
df_renamed = df_transformed.withColumnRenamed("unit_price", "price")

# Drop columns
df_dropped = df_transformed.drop("day_of_week")

# Cast types
df_cast = sales_df.withColumn("quantity", F.col("quantity").cast(DoubleType()))

# --- 3.2 Conditional Logic ---

# CASE WHEN equivalent
df_with_category = sales_df.withColumn(
    "price_tier",
    F.when(F.col("unit_price") < 20, "Budget")
     .when(F.col("unit_price") < 40, "Standard")
     .otherwise("Premium")
)
df_with_category.show()

# Multiple conditions
df_flagged = sales_df.withColumn(
    "high_volume_premium",
    F.when(
        (F.col("quantity") > 100) & (F.col("unit_price") > 30),
        True
    ).otherwise(False)
)
df_flagged.show()

# --- 3.3 String Operations ---

df_strings = spark.createDataFrame([
    ("  ALICE SMITH  ", "alice@email.com"),
    ("bob jones", "BOB@EMAIL.COM"),
    ("  Charlie Brown  ", "charlie@email.com")
], ["name", "email"])

df_cleaned = df_strings \
    .withColumn("name_clean", F.trim(F.initcap(F.col("name")))) \
    .withColumn("email_lower", F.lower(F.col("email"))) \
    .withColumn("email_domain", F.split(F.col("email"), "@").getItem(1)) \
    .withColumn("name_length", F.length(F.col("name")))

df_cleaned.show(truncate=False)

# EXERCISE 3.1: Using sales_df, create a transformation that:
# - Adds a 'revenue' column (quantity * unit_price)
# - Adds a 'is_weekend' column (Saturday=7, Sunday=1 in dayofweek)
# - Adds a 'product_upper' column with product name in uppercase
# - Filters only rows where revenue > 2000
# YOUR CODE HERE:
# result = ...


# =============================================================================
# SECTION 4: Aggregations and GroupBy
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Aggregations and GroupBy")
print("=" * 70)

# --- 4.1 Basic Aggregations ---

# Single aggregations
print("Total quantity sold:")
sales_df.select(F.sum("quantity")).show()

# Multiple aggregations
print("\nMultiple aggregations:")
sales_df.select(
    F.count("*").alias("total_rows"),
    F.sum("quantity").alias("total_quantity"),
    F.avg("unit_price").alias("avg_price"),
    F.min("unit_price").alias("min_price"),
    F.max("unit_price").alias("max_price"),
    F.countDistinct("product").alias("unique_products")
).show()

# --- 4.2 GroupBy Operations ---

# Group by single column
print("\nSales by product:")
sales_df.groupBy("product") \
    .agg(
        F.sum("quantity").alias("total_quantity"),
        F.sum(F.col("quantity") * F.col("unit_price")).alias("total_revenue"),
        F.count("*").alias("num_transactions")
    ) \
    .orderBy(F.desc("total_revenue")) \
    .show()

# Group by multiple columns
print("\nSales by region and category:")
sales_df.groupBy("region", "category") \
    .agg(
        F.sum("quantity").alias("total_quantity"),
        F.avg("unit_price").alias("avg_price")
    ) \
    .orderBy("region", "category") \
    .show()

# --- 4.3 Pivot Tables ---

print("\nPivot - Quantity by Region and Product:")
pivot_df = sales_df.groupBy("region") \
    .pivot("product") \
    .agg(F.sum("quantity"))
pivot_df.show()

# --- 4.4 Rollup and Cube (for OLAP) ---

# Rollup - hierarchical aggregation
print("\nRollup (Category -> Product):")
sales_df.rollup("category", "product") \
    .agg(F.sum("quantity").alias("total_qty")) \
    .orderBy("category", "product") \
    .show()

# EXERCISE 4.1: Calculate the following metrics by region:
# - Total revenue
# - Average quantity per transaction
# - Number of unique products sold
# - Most common product (mode) - hint: use groupBy twice
# YOUR CODE HERE:
# region_stats = ...


# =============================================================================
# SECTION 5: Window Functions - Interview Essential!
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Window Functions")
print("=" * 70)

# Create transaction data for window examples
transactions = spark.createDataFrame([
    (1, "2024-01-01", 100.0),
    (1, "2024-01-02", 150.0),
    (1, "2024-01-03", 200.0),
    (1, "2024-01-04", 120.0),
    (2, "2024-01-01", 300.0),
    (2, "2024-01-02", 250.0),
    (2, "2024-01-03", 400.0),
    (3, "2024-01-01", 500.0),
    (3, "2024-01-02", 450.0),
], ["user_id", "date", "amount"])

transactions = transactions.withColumn("date", F.to_date("date"))

# --- 5.1 Ranking Functions ---

# Define window specification
window_by_user = Window.partitionBy("user_id").orderBy(F.desc("amount"))

ranked = transactions.withColumn("rank", F.rank().over(window_by_user)) \
    .withColumn("dense_rank", F.dense_rank().over(window_by_user)) \
    .withColumn("row_number", F.row_number().over(window_by_user))

print("Ranking within each user:")
ranked.show()

# --- 5.2 Running Totals and Moving Averages ---

window_running = Window.partitionBy("user_id").orderBy("date") \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

window_moving = Window.partitionBy("user_id").orderBy("date") \
    .rowsBetween(-2, Window.currentRow)  # Last 3 rows including current

running_stats = transactions \
    .withColumn("cumulative_sum", F.sum("amount").over(window_running)) \
    .withColumn("cumulative_avg", F.avg("amount").over(window_running)) \
    .withColumn("moving_avg_3", F.avg("amount").over(window_moving))

print("\nRunning totals and moving averages:")
running_stats.show()

# --- 5.3 LAG and LEAD ---

window_order = Window.partitionBy("user_id").orderBy("date")

lag_lead = transactions \
    .withColumn("prev_amount", F.lag("amount", 1).over(window_order)) \
    .withColumn("next_amount", F.lead("amount", 1).over(window_order)) \
    .withColumn("amount_change", F.col("amount") - F.lag("amount", 1).over(window_order)) \
    .withColumn("pct_change",
        (F.col("amount") - F.lag("amount", 1).over(window_order)) /
        F.lag("amount", 1).over(window_order) * 100
    )

print("\nLAG and LEAD:")
lag_lead.show()

# --- 5.4 FIRST_VALUE, LAST_VALUE, NTH_VALUE ---

window_full = Window.partitionBy("user_id").orderBy("date") \
    .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

first_last = transactions \
    .withColumn("first_amount", F.first("amount").over(window_full)) \
    .withColumn("last_amount", F.last("amount").over(window_full)) \
    .withColumn("max_amount", F.max("amount").over(window_full))

print("\nFIRST and LAST values:")
first_last.show()

# --- 5.5 Percent Rank and NTile ---

window_all = Window.orderBy("amount")

percentiles = transactions \
    .withColumn("percent_rank", F.percent_rank().over(window_all)) \
    .withColumn("ntile_4", F.ntile(4).over(window_all))

print("\nPercentiles:")
percentiles.show()

# EXERCISE 5.1: Using transactions data, calculate for each user:
# - Their highest transaction amount
# - Days since their first transaction
# - Running count of transactions
# - Whether current amount is above their average (boolean)
# YOUR CODE HERE:
# user_analysis = ...


# =============================================================================
# SECTION 6: Joins
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6: Joins")
print("=" * 70)

# Create dimension tables
users = spark.createDataFrame([
    (1, "Alice", "Gold"),
    (2, "Bob", "Silver"),
    (3, "Charlie", "Bronze"),
    (4, "Diana", "Gold"),  # No transactions
], ["user_id", "name", "tier"])

products = spark.createDataFrame([
    ("Widget", "Electronics", "Supplier A"),
    ("Gadget", "Electronics", "Supplier B"),
    ("Tool", "Hardware", "Supplier A"),
    ("Accessory", "Electronics", "Supplier C"),
    ("Unused", "Other", "Supplier D"),  # Never sold
], ["product", "category", "supplier"])

# --- 6.1 Inner Join ---
print("Inner Join (only matching rows):")
inner_join = transactions.join(users, "user_id", "inner")
inner_join.show()

# --- 6.2 Left Join ---
print("\nLeft Join (all from left, matching from right):")
left_join = users.join(transactions, "user_id", "left")
left_join.show()

# --- 6.3 Right Join ---
print("\nRight Join:")
right_join = transactions.join(users, "user_id", "right")
right_join.show()

# --- 6.4 Full Outer Join ---
print("\nFull Outer Join:")
full_join = transactions.join(users, "user_id", "full")
full_join.show()

# --- 6.5 Left Anti Join (NOT IN equivalent) ---
print("\nLeft Anti Join (users WITHOUT transactions):")
no_transactions = users.join(transactions, "user_id", "left_anti")
no_transactions.show()

# --- 6.6 Left Semi Join (EXISTS equivalent) ---
print("\nLeft Semi Join (users WITH transactions):")
has_transactions = users.join(transactions, "user_id", "left_semi")
has_transactions.show()

# --- 6.7 Cross Join (Cartesian Product) ---
print("\nCross Join (use carefully!):")
# cross_join = users.crossJoin(products)  # Can be huge!

# --- 6.8 Join with Different Column Names ---
orders = spark.createDataFrame([
    (1, 101, 2),
    (2, 102, 1),
], ["order_id", "customer_id", "quantity"])

customers = spark.createDataFrame([
    (101, "Alice"),
    (102, "Bob"),
], ["id", "name"])

# Join on differently named columns
joined = orders.join(
    customers,
    orders.customer_id == customers.id,
    "inner"
).drop("id")  # Drop duplicate column

print("\nJoin with different column names:")
joined.show()

# EXERCISE 6.1:
# - Join sales_df with products on 'product' column
# - Find products that have never been sold (anti join)
# - Calculate total revenue by supplier
# YOUR CODE HERE:
# sales_with_supplier = ...
# unsold_products = ...
# revenue_by_supplier = ...


# =============================================================================
# SECTION 7: Spark SQL
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 7: Spark SQL")
print("=" * 70)

# Register DataFrames as temporary views
sales_df.createOrReplaceTempView("sales")
users.createOrReplaceTempView("users")
transactions.createOrReplaceTempView("transactions")

# --- 7.1 Basic SQL Queries ---

print("SQL Query - Sales by product:")
spark.sql("""
    SELECT
        product,
        SUM(quantity) as total_quantity,
        SUM(quantity * unit_price) as total_revenue
    FROM sales
    GROUP BY product
    ORDER BY total_revenue DESC
""").show()

# --- 7.2 Window Functions in SQL ---

print("\nSQL Window Functions:")
spark.sql("""
    SELECT
        user_id,
        date,
        amount,
        SUM(amount) OVER (
            PARTITION BY user_id
            ORDER BY date
            ROWS UNBOUNDED PRECEDING
        ) as cumulative_amount,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY amount DESC
        ) as amount_rank
    FROM transactions
""").show()

# --- 7.3 CTEs in Spark SQL ---

print("\nSQL with CTEs:")
spark.sql("""
    WITH user_stats AS (
        SELECT
            user_id,
            COUNT(*) as transaction_count,
            SUM(amount) as total_amount,
            AVG(amount) as avg_amount
        FROM transactions
        GROUP BY user_id
    ),
    ranked_users AS (
        SELECT
            *,
            RANK() OVER (ORDER BY total_amount DESC) as spending_rank
        FROM user_stats
    )
    SELECT * FROM ranked_users
""").show()

# --- 7.4 Complex SQL Queries ---

print("\nComplex SQL - RFM Analysis:")
spark.sql("""
    SELECT
        user_id,
        DATEDIFF('2024-01-05', MAX(date)) as recency_days,
        COUNT(*) as frequency,
        SUM(amount) as monetary,
        CASE
            WHEN SUM(amount) > 500 THEN 'High Value'
            WHEN SUM(amount) > 200 THEN 'Medium Value'
            ELSE 'Low Value'
        END as value_segment
    FROM transactions
    GROUP BY user_id
    ORDER BY monetary DESC
""").show()


# =============================================================================
# SECTION 8: Performance Optimization
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 8: Performance Optimization")
print("=" * 70)

# --- 8.1 Caching ---

# Cache frequently used DataFrames
sales_df.cache()  # or .persist()

# Unpersist when done
# sales_df.unpersist()

# --- 8.2 Repartitioning ---

# Check current partitions
print(f"Current partitions: {sales_df.rdd.getNumPartitions()}")

# Repartition (full shuffle - expensive)
df_repartitioned = sales_df.repartition(4)

# Repartition by column (for joins/aggregations)
df_by_region = sales_df.repartition("region")

# Coalesce (reduce partitions - no shuffle, cheaper)
df_coalesced = sales_df.coalesce(2)

# --- 8.3 Broadcast Joins ---

from pyspark.sql.functions import broadcast

# Small table should be broadcast
# joined = large_df.join(broadcast(small_df), "key")

print("\nBroadcast join (small table to all nodes):")
result = sales_df.join(broadcast(products), "product")
result.explain()  # Shows BroadcastHashJoin

# --- 8.4 Avoiding Shuffles ---

print("""
PERFORMANCE TIPS:

1. Use broadcast joins for small tables (< 10MB)
2. Partition data by frequently filtered/joined columns
3. Use coalesce instead of repartition when reducing partitions
4. Cache intermediate results that are reused
5. Filter early to reduce data volume
6. Select only needed columns early
7. Avoid UDFs when built-in functions exist (UDFs are slower)
8. Use Adaptive Query Execution (AQE) in Spark 3.x

Key configs:
- spark.sql.shuffle.partitions (default 200, adjust based on data size)
- spark.sql.adaptive.enabled = true
- spark.sql.autoBroadcastJoinThreshold (default 10MB)
""")

# --- 8.5 Explain Plan ---

print("\nQuery Execution Plan:")
sales_df.filter(F.col("category") == "Electronics") \
    .groupBy("product") \
    .agg(F.sum("quantity")) \
    .explain(mode="formatted")


# =============================================================================
# SECTION 9: UDFs and Complex Types
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 9: UDFs and Complex Types")
print("=" * 70)

# --- 9.1 User Defined Functions (UDFs) ---

from pyspark.sql.functions import udf, pandas_udf
from pyspark.sql.types import StringType, ArrayType

# Simple UDF
@udf(returnType=StringType())
def categorize_amount(amount):
    if amount is None:
        return "Unknown"
    elif amount < 100:
        return "Low"
    elif amount < 300:
        return "Medium"
    else:
        return "High"

print("UDF Example:")
transactions.withColumn("amount_category", categorize_amount(F.col("amount"))).show()

# --- 9.2 Pandas UDFs (Vectorized - MUCH FASTER) ---

# Pandas UDF for better performance
@pandas_udf(StringType())
def categorize_amount_pandas(amounts):
    import pandas as pd
    return pd.cut(
        amounts,
        bins=[0, 100, 300, float('inf')],
        labels=['Low', 'Medium', 'High']
    ).astype(str)

print("\nPandas UDF (faster):")
transactions.withColumn("amount_category", categorize_amount_pandas(F.col("amount"))).show()

# --- 9.3 Working with Arrays ---

array_df = spark.createDataFrame([
    (1, ["apple", "banana", "orange"]),
    (2, ["grape", "melon"]),
    (3, ["apple", "grape", "kiwi"]),
], ["id", "fruits"])

print("\nArray operations:")
array_df \
    .withColumn("num_fruits", F.size("fruits")) \
    .withColumn("has_apple", F.array_contains("fruits", "apple")) \
    .withColumn("first_fruit", F.element_at("fruits", 1)) \
    .withColumn("fruits_sorted", F.sort_array("fruits")) \
    .show(truncate=False)

# Explode array to rows
print("\nExplode array:")
array_df.select("id", F.explode("fruits").alias("fruit")).show()

# --- 9.4 Working with Maps ---

map_df = spark.createDataFrame([
    (1, {"name": "Alice", "city": "NYC"}),
    (2, {"name": "Bob", "city": "LA", "country": "USA"}),
], ["id", "info"])

print("\nMap operations:")
map_df \
    .withColumn("name", F.col("info").getItem("name")) \
    .withColumn("keys", F.map_keys("info")) \
    .withColumn("values", F.map_values("info")) \
    .show(truncate=False)

# --- 9.5 Working with Structs ---

struct_df = spark.createDataFrame([
    (1, ("Alice", 25)),
    (2, ("Bob", 30)),
], ["id", "person"]) \
    .select(
        "id",
        F.col("person._1").alias("name"),
        F.col("person._2").alias("age")
    )

print("\nStruct operations:")
struct_df.show()

# Create struct
df_with_struct = df.select(
    F.struct("name", "age", "salary").alias("employee_info"),
    "department"
)
df_with_struct.show(truncate=False)


# =============================================================================
# SECTION 10: Production Patterns
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 10: Production Patterns")
print("=" * 70)

# --- 10.1 Schema Validation ---

def validate_schema(df, expected_schema):
    """Validate DataFrame schema matches expected."""
    actual_fields = {f.name: f.dataType for f in df.schema.fields}
    expected_fields = {f.name: f.dataType for f in expected_schema.fields}

    missing = set(expected_fields.keys()) - set(actual_fields.keys())
    unexpected = set(actual_fields.keys()) - set(expected_fields.keys())
    type_mismatches = {
        k: (expected_fields[k], actual_fields[k])
        for k in expected_fields.keys() & actual_fields.keys()
        if expected_fields[k] != actual_fields[k]
    }

    return {
        'valid': not (missing or type_mismatches),
        'missing_columns': missing,
        'unexpected_columns': unexpected,
        'type_mismatches': type_mismatches
    }

print("Schema Validation:")
validation = validate_schema(sales_df, sales_schema)
print(validation)

# --- 10.2 Data Quality Checks ---

def run_quality_checks(df, checks):
    """
    Run data quality checks.
    checks: dict of {check_name: (column, condition_func)}
    """
    results = {}
    total_rows = df.count()

    for check_name, (column, condition) in checks.items():
        failing_rows = df.filter(~condition(F.col(column))).count()
        results[check_name] = {
            'passing': total_rows - failing_rows,
            'failing': failing_rows,
            'pass_rate': (total_rows - failing_rows) / total_rows * 100
        }

    return results

quality_checks = {
    'quantity_positive': ('quantity', lambda c: c > 0),
    'price_valid': ('unit_price', lambda c: (c > 0) & (c < 1000)),
    'product_not_null': ('product', lambda c: c.isNotNull()),
}

print("\nData Quality Results:")
quality_results = run_quality_checks(sales_df, quality_checks)
for check, result in quality_results.items():
    print(f"  {check}: {result['pass_rate']:.1f}% passing")

# --- 10.3 Incremental Processing Pattern ---

def process_incremental(
    spark,
    source_path,
    target_path,
    watermark_path,
    key_column
):
    """
    Process only new/changed records since last run.
    """
    # Read watermark (last processed timestamp)
    # watermark = spark.read.text(watermark_path).first()[0]

    # Read only new data
    # new_data = spark.read.parquet(source_path) \
    #     .filter(F.col("updated_at") > watermark)

    # Read existing data
    # existing = spark.read.parquet(target_path)

    # Merge (upsert)
    # merged = existing.join(new_data, key_column, "left_anti") \
    #     .union(new_data)

    # Write and update watermark
    # merged.write.mode("overwrite").parquet(target_path)
    pass

# --- 10.4 Slowly Changing Dimension Type 2 ---

def scd_type2_merge(current_df, updates_df, key_columns, track_columns):
    """
    Implement SCD Type 2 merge logic.
    """
    # Mark current records that will be updated
    # Close old records (set end_date, is_current=False)
    # Insert new versions
    pass

# --- 10.5 Handling Late Arriving Data ---

def handle_late_data(df, event_time_col, watermark_delay="1 hour"):
    """
    For streaming: handle late arriving events.
    """
    # In structured streaming:
    # df.withWatermark(event_time_col, watermark_delay)
    pass


# =============================================================================
# SECTION 11: Interview Questions & Exercises
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 11: Interview Exercises")
print("=" * 70)

print("""
=== EXERCISE 11.1: Top N per Group ===
Find the top 2 transactions by amount for each user.
Use window functions with row_number.
""")

# Solution
window = Window.partitionBy("user_id").orderBy(F.desc("amount"))
top_2_per_user = transactions \
    .withColumn("rank", F.row_number().over(window)) \
    .filter(F.col("rank") <= 2) \
    .drop("rank")

print("Top 2 transactions per user:")
top_2_per_user.show()


print("""
=== EXERCISE 11.2: Running Difference ===
Calculate the difference between current and previous transaction for each user.
""")

# Solution
window = Window.partitionBy("user_id").orderBy("date")
running_diff = transactions \
    .withColumn("prev_amount", F.lag("amount").over(window)) \
    .withColumn("diff", F.col("amount") - F.col("prev_amount")) \
    .withColumn("pct_change",
        F.when(F.col("prev_amount").isNotNull(),
            F.round((F.col("amount") - F.col("prev_amount")) / F.col("prev_amount") * 100, 2)
        )
    )

print("Running difference:")
running_diff.show()


print("""
=== EXERCISE 11.3: Sessionization ===
Group events into sessions (new session if gap > 30 minutes).
""")

# Create event data
events = spark.createDataFrame([
    (1, "2024-01-01 10:00:00", "page_view"),
    (1, "2024-01-01 10:05:00", "click"),
    (1, "2024-01-01 10:08:00", "purchase"),
    (1, "2024-01-01 11:00:00", "page_view"),  # New session
    (1, "2024-01-01 11:05:00", "click"),
    (2, "2024-01-01 09:00:00", "page_view"),
    (2, "2024-01-01 09:10:00", "click"),
], ["user_id", "event_time", "event_type"])

events = events.withColumn("event_time", F.to_timestamp("event_time"))

# Solution
window = Window.partitionBy("user_id").orderBy("event_time")

sessionized = events \
    .withColumn("prev_time", F.lag("event_time").over(window)) \
    .withColumn("time_diff_minutes",
        (F.unix_timestamp("event_time") - F.unix_timestamp("prev_time")) / 60
    ) \
    .withColumn("is_new_session",
        F.when(
            F.col("prev_time").isNull() | (F.col("time_diff_minutes") > 30),
            1
        ).otherwise(0)
    ) \
    .withColumn("session_id",
        F.sum("is_new_session").over(window)
    )

print("Sessionized events:")
sessionized.select("user_id", "event_time", "event_type", "session_id").show()


print("""
=== EXERCISE 11.4: Find Gaps in Sequence ===
Find missing IDs in a sequence.
""")

# Create data with gaps
ids_df = spark.createDataFrame([(1,), (2,), (4,), (7,), (8,), (10,)], ["id"])

# Solution
window = Window.orderBy("id")
gaps = ids_df \
    .withColumn("prev_id", F.lag("id").over(window)) \
    .withColumn("gap", F.col("id") - F.col("prev_id")) \
    .filter(F.col("gap") > 1) \
    .withColumn("missing_start", F.col("prev_id") + 1) \
    .withColumn("missing_end", F.col("id") - 1)

print("Gaps found:")
gaps.select("prev_id", "id", "missing_start", "missing_end").show()


print("""
=== EXERCISE 11.5: Pivoting and Unpivoting ===
Convert wide format to long format and vice versa.
""")

# Wide format
wide_df = spark.createDataFrame([
    (1, 100, 150, 200),
    (2, 80, 120, 90),
], ["user_id", "jan_sales", "feb_sales", "mar_sales"])

# Unpivot (melt) - wide to long
unpivoted = wide_df.selectExpr(
    "user_id",
    "stack(3, 'jan', jan_sales, 'feb', feb_sales, 'mar', mar_sales) as (month, sales)"
)

print("Unpivoted (wide to long):")
unpivoted.show()

# Pivot - long to wide
pivoted = unpivoted.groupBy("user_id").pivot("month").sum("sales")
print("Pivoted (long to wide):")
pivoted.show()


# =============================================================================
# CLEANUP
# =============================================================================

print("\n" + "=" * 70)
print("TUTORIAL COMPLETE!")
print("=" * 70)

print("""
Key Takeaways:

1. ALWAYS define schemas explicitly in production
2. Use window functions for row-level comparisons within groups
3. Broadcast small tables for efficient joins
4. Filter and select early to reduce data volume
5. Prefer built-in functions over UDFs (performance)
6. Use Pandas UDFs when UDFs are necessary
7. Cache/persist intermediate results that are reused
8. Monitor partitioning - too few = memory issues, too many = overhead
9. Use explain() to understand query execution
10. Enable Adaptive Query Execution in Spark 3.x

Common Interview Topics:
- Window functions (ranking, running totals, lag/lead)
- Joins (especially broadcast and anti joins)
- Partitioning strategies
- Handling skewed data
- Optimizing shuffle operations
- Incremental processing patterns
""")

# Stop SparkSession
# spark.stop()  # Uncomment in production scripts
