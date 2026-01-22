"""
=============================================================================
PySpark + Databricks Tutorial - Leveraging Enterprise Infrastructure
=============================================================================
Focus: You already have Spark/Databricks at your company - use it to your advantage!

This tutorial assumes:
- Your company has Databricks, EMR, or Dataproc
- You want to leverage existing infrastructure
- You want to stand out as a Data/ML Engineer

WHY THIS MATTERS FOR YOUR CAREER:
- Companies pay $$$ for Spark clusters - they WANT you to use them
- Knowing Databricks-specific features makes you more valuable
- You can process data that others can't (too big for pandas/polars)
- Enterprise features = production-ready code = promotions
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, DateType, TimestampType, ArrayType
)
from datetime import datetime, timedelta
import os

# =============================================================================
# SECTION 1: Leveraging Your Company's Spark Cluster
# =============================================================================

print("=" * 70)
print("SECTION 1: Leveraging Enterprise Spark Infrastructure")
print("=" * 70)

# --- 1.1 Local Development (this tutorial) ---
spark = SparkSession.builder \
    .appName("DatabricksTutorial") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

# --- 1.2 Databricks (what you'd use at work) ---
"""
In Databricks, SparkSession is pre-configured as 'spark'
You get these FOR FREE:

# Already available - no setup needed!
spark  # SparkSession
dbutils  # Databricks utilities
display()  # Rich DataFrame visualization

# Connect to cloud storage (already configured)
df = spark.read.parquet("s3://company-data-lake/events/")
df = spark.read.parquet("abfss://container@storage.dfs.core.windows.net/data/")
df = spark.read.parquet("gs://company-bucket/data/")
"""

print("""
=== WHAT YOUR COMPANY'S CLUSTER GIVES YOU ===

1. COMPUTE POWER
   - 100s of cores vs your laptop's 4-8
   - TBs of RAM vs your laptop's 16-32GB
   - Process in minutes what takes hours locally

2. DATA ACCESS
   - Direct connection to data lakes (S3, ADLS, GCS)
   - No need to download data locally
   - Access to production databases

3. ENTERPRISE FEATURES
   - Unity Catalog (data governance)
   - Delta Lake (ACID transactions)
   - MLflow (model tracking)
   - Job scheduling

4. COLLABORATION
   - Shared notebooks
   - Version control integration
   - Team workspaces
""")


# =============================================================================
# SECTION 2: Delta Lake - Your Secret Weapon
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: Delta Lake - ACID Transactions for Big Data")
print("=" * 70)

"""
Delta Lake is likely already enabled in your Databricks workspace.
This is a HUGE advantage over plain Parquet files.

=== WHY DELTA LAKE MATTERS ===

1. ACID Transactions - No corrupted data from failed jobs
2. Time Travel - Query data as it was in the past
3. Schema Evolution - Add columns without breaking pipelines
4. Audit History - Track all changes
5. MERGE (Upsert) - Update + Insert in one operation
"""

# Simulated Delta operations (would work in Databricks)
print("""
# --- DELTA LAKE OPERATIONS (Databricks) ---

# Read Delta table
df = spark.read.format("delta").load("/mnt/delta/events")

# Or use table name directly (Unity Catalog)
df = spark.table("catalog.schema.events")

# Write as Delta (creates table if not exists)
df.write.format("delta") \\
    .mode("overwrite") \\
    .partitionBy("date") \\
    .save("/mnt/delta/processed_events")

# TIME TRAVEL - Query historical data!
# As of timestamp
df_yesterday = spark.read.format("delta") \\
    .option("timestampAsOf", "2024-01-15") \\
    .load("/mnt/delta/events")

# As of version
df_v5 = spark.read.format("delta") \\
    .option("versionAsOf", 5) \\
    .load("/mnt/delta/events")

# MERGE (Upsert) - The killer feature
# Updates existing records, inserts new ones
from delta.tables import DeltaTable

delta_table = DeltaTable.forPath(spark, "/mnt/delta/customers")
updates_df = spark.read.parquet("/mnt/incoming/customer_updates")

delta_table.alias("target").merge(
    updates_df.alias("source"),
    "target.customer_id = source.customer_id"
).whenMatchedUpdateAll() \\
 .whenNotMatchedInsertAll() \\
 .execute()

# VACUUM - Clean up old files (save storage costs!)
delta_table.vacuum(168)  # Keep 7 days of history
""")


# =============================================================================
# SECTION 3: Databricks-Specific Features
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Databricks-Specific Features")
print("=" * 70)

print("""
=== DBUTILS - Your Swiss Army Knife ===

# File operations
dbutils.fs.ls("/mnt/data/")
dbutils.fs.cp("/source/file.parquet", "/dest/file.parquet")
dbutils.fs.rm("/old/data/", recurse=True)

# Secrets management (for credentials)
password = dbutils.secrets.get(scope="my-scope", key="db-password")

# Widgets (parameterize notebooks)
dbutils.widgets.text("start_date", "2024-01-01")
dbutils.widgets.dropdown("environment", "dev", ["dev", "staging", "prod"])
start_date = dbutils.widgets.get("start_date")

# Notebook workflows
dbutils.notebook.run("/path/to/notebook", timeout_seconds=3600, arguments={"date": "2024-01-01"})

# Exit with value (for job orchestration)
dbutils.notebook.exit("SUCCESS")


=== DISPLAY() - Rich Visualization ===

# Instead of df.show(), use:
display(df)  # Interactive table with sorting, filtering
display(df.groupBy("category").count())  # Auto-generates charts!


=== %sql MAGIC - Write SQL Directly ===

-- In a notebook cell, you can write pure SQL:
%sql
SELECT
    category,
    COUNT(*) as count,
    AVG(amount) as avg_amount
FROM events
WHERE date >= '2024-01-01'
GROUP BY category
ORDER BY count DESC


=== AUTOLOADER - Incremental Data Ingestion ===

# Automatically process new files as they arrive
df = spark.readStream.format("cloudFiles") \\
    .option("cloudFiles.format", "json") \\
    .option("cloudFiles.schemaLocation", "/mnt/schema/events") \\
    .load("/mnt/raw/events/")

df.writeStream \\
    .format("delta") \\
    .option("checkpointLocation", "/mnt/checkpoints/events") \\
    .trigger(availableNow=True) \\
    .toTable("catalog.schema.events")
""")


# =============================================================================
# SECTION 4: Production Patterns with Delta Lake
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Production Patterns")
print("=" * 70)

# Create sample data for demonstrations
events_data = [
    (1, "2024-01-15 10:00:00", "page_view", "homepage", 1.5),
    (1, "2024-01-15 10:01:00", "click", "product_123", 0.0),
    (1, "2024-01-15 10:02:00", "add_to_cart", "product_123", 29.99),
    (1, "2024-01-15 10:05:00", "purchase", "product_123", 29.99),
    (2, "2024-01-15 11:00:00", "page_view", "homepage", 2.0),
    (2, "2024-01-15 11:03:00", "search", "laptops", 0.0),
    (2, "2024-01-15 11:05:00", "page_view", "product_456", 3.5),
    (3, "2024-01-16 09:00:00", "page_view", "homepage", 1.0),
    (3, "2024-01-16 09:01:00", "click", "banner_promo", 0.0),
]

events_schema = StructType([
    StructField("user_id", IntegerType()),
    StructField("event_time", StringType()),
    StructField("event_type", StringType()),
    StructField("page", StringType()),
    StructField("value", DoubleType())
])

events_df = spark.createDataFrame(events_data, events_schema)
events_df = events_df.withColumn("event_time", F.to_timestamp("event_time"))
events_df.createOrReplaceTempView("events")

# --- 4.1 Medallion Architecture (Bronze/Silver/Gold) ---
print("""
=== MEDALLION ARCHITECTURE ===
The standard pattern for data lakes:

┌─────────┐     ┌─────────┐     ┌─────────┐
│ BRONZE  │ --> │ SILVER  │ --> │  GOLD   │
│  (Raw)  │     │(Cleaned)│     │ (Curated│
└─────────┘     └─────────┘     └─────────┘

BRONZE: Raw data as-is from sources
- Append-only
- Keep original schema
- Minimal transformations

SILVER: Cleaned and validated
- Deduplication
- Schema enforcement
- Data quality checks

GOLD: Business-ready aggregations
- Pre-computed metrics
- Optimized for queries
- Serves dashboards/ML
""")

# Bronze -> Silver transformation example
print("\n--- Bronze to Silver: Cleaning and Validation ---")

# Simulate bronze data with quality issues
bronze_data = [
    (1, "2024-01-15", "purchase", 100.0, "valid"),
    (1, "2024-01-15", "purchase", 100.0, "duplicate"),  # Duplicate
    (2, "invalid_date", "click", 0.0, "bad_date"),      # Bad date
    (3, "2024-01-16", "view", -50.0, "negative"),       # Negative value
    (4, "2024-01-16", "purchase", 200.0, "valid"),
]

bronze_df = spark.createDataFrame(
    bronze_data,
    ["user_id", "date", "event_type", "amount", "note"]
)

# Silver transformation pipeline
silver_df = (
    bronze_df
    # Remove duplicates
    .dropDuplicates(["user_id", "date", "event_type", "amount"])
    # Validate and parse dates
    .withColumn("date_parsed", F.to_date("date", "yyyy-MM-dd"))
    .filter(F.col("date_parsed").isNotNull())
    # Filter invalid values
    .filter(F.col("amount") >= 0)
    # Add audit columns
    .withColumn("processed_at", F.current_timestamp())
    .withColumn("source_file", F.lit("bronze_events_20240115"))
)

print("Bronze data:")
bronze_df.show()
print("Silver data (cleaned):")
silver_df.show()

# Silver -> Gold aggregation
print("\n--- Silver to Gold: Business Metrics ---")

gold_daily_metrics = (
    silver_df
    .groupBy("date_parsed")
    .agg(
        F.countDistinct("user_id").alias("unique_users"),
        F.count("*").alias("total_events"),
        F.sum("amount").alias("total_revenue"),
        F.avg("amount").alias("avg_transaction")
    )
)

print("Gold metrics:")
gold_daily_metrics.show()


# --- 4.2 Incremental Processing Pattern ---
print("""
=== INCREMENTAL PROCESSING ===
Don't reprocess everything - only new data!

# Using watermarks (timestamp-based)
def process_incremental(spark, source_path, target_table, watermark_table):
    # Get last processed timestamp
    last_watermark = spark.sql(f'''
        SELECT MAX(processed_until) as watermark
        FROM {watermark_table}
    ''').collect()[0]['watermark'] or '1970-01-01'

    # Read only new data
    new_data = spark.read.parquet(source_path) \\
        .filter(F.col("event_time") > last_watermark)

    if new_data.count() > 0:
        # Process and write
        processed = transform(new_data)
        processed.write.format("delta") \\
            .mode("append") \\
            .saveAsTable(target_table)

        # Update watermark
        new_watermark = new_data.agg(F.max("event_time")).collect()[0][0]
        spark.sql(f'''
            INSERT INTO {watermark_table} VALUES ('{new_watermark}', current_timestamp())
        ''')
""")


# --- 4.3 SCD Type 2 with Delta Lake ---
print("""
=== SCD TYPE 2 - Track Historical Changes ===

# Slowly Changing Dimension Type 2 implementation
from delta.tables import DeltaTable

def scd2_merge(spark, updates_df, target_table, key_columns, track_columns):
    '''
    Implements SCD Type 2:
    - Close old records (set end_date, is_current=False)
    - Insert new versions
    '''
    target = DeltaTable.forName(spark, target_table)

    # Prepare merge condition
    key_condition = " AND ".join([f"target.{k} = source.{k}" for k in key_columns])

    # Find changed records
    change_condition = " OR ".join([f"target.{c} != source.{c}" for c in track_columns])

    # Step 1: Update existing records (close them)
    target.alias("target").merge(
        updates_df.alias("source"),
        key_condition
    ).whenMatchedUpdate(
        condition=f"target.is_current = true AND ({change_condition})",
        set={
            "end_date": F.current_date(),
            "is_current": F.lit(False)
        }
    ).execute()

    # Step 2: Insert new versions
    new_records = updates_df \\
        .withColumn("start_date", F.current_date()) \\
        .withColumn("end_date", F.lit(None).cast("date")) \\
        .withColumn("is_current", F.lit(True))

    # Insert only changed/new records
    new_records.write.format("delta") \\
        .mode("append") \\
        .saveAsTable(target_table)

# Example customer dimension
# | customer_id | name  | tier   | start_date | end_date   | is_current |
# |-------------|-------|--------|------------|------------|------------|
# | 1           | Alice | Silver | 2024-01-01 | 2024-03-15 | false      |
# | 1           | Alice | Gold   | 2024-03-15 | null       | true       |
""")


# =============================================================================
# SECTION 5: Performance Optimization (Enterprise Scale)
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Performance Optimization at Scale")
print("=" * 70)

print("""
=== OPTIMIZATION TECHNIQUES FOR ENTERPRISE SPARK ===

1. PARTITIONING - Organize data for efficient queries

   # Partition by date (most common)
   df.write.format("delta") \\
       .partitionBy("year", "month", "day") \\
       .save("/mnt/delta/events")

   # Query only reads relevant partitions
   spark.read.format("delta").load("/mnt/delta/events") \\
       .filter("year = 2024 AND month = 1")  # Only reads Jan 2024!

2. Z-ORDERING - Colocate related data (Delta Lake)

   # Optimize for common filter columns
   OPTIMIZE delta.`/mnt/delta/events`
   ZORDER BY (user_id, event_type)

   # Queries filtering on user_id or event_type are MUCH faster

3. CACHING - Keep hot data in memory

   # Cache DataFrame that's reused
   df.cache()  # or df.persist()

   # Cache Delta table
   %sql
   CACHE SELECT * FROM events WHERE date >= '2024-01-01'

4. BROADCAST JOINS - Small table to all nodes

   from pyspark.sql.functions import broadcast

   # Broadcast dimension tables (< 10MB)
   result = large_fact_table.join(
       broadcast(small_dim_table),
       "key"
   )

5. ADAPTIVE QUERY EXECUTION (AQE) - Auto-optimization

   # Usually enabled by default in Databricks
   spark.conf.set("spark.sql.adaptive.enabled", "true")
   spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
   spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

6. PHOTON ENGINE - Databricks native vectorized engine

   # Enable in cluster settings - up to 3x faster!
   # No code changes needed
""")

# Demonstrate broadcast join
print("\n--- Broadcast Join Example ---")

# Large fact table (transactions)
transactions = spark.createDataFrame([
    (1, 101, 100.0),
    (2, 102, 200.0),
    (3, 101, 150.0),
    (4, 103, 300.0),
] * 1000, ["transaction_id", "product_id", "amount"])

# Small dimension table (products)
products = spark.createDataFrame([
    (101, "Widget", "Electronics"),
    (102, "Gadget", "Electronics"),
    (103, "Tool", "Hardware"),
], ["product_id", "name", "category"])

# Broadcast the small table
result = transactions.join(
    F.broadcast(products),
    "product_id"
)

print("Execution plan with broadcast:")
result.explain()


# =============================================================================
# SECTION 6: ML Integration with MLflow
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6: ML Integration (MLflow)")
print("=" * 70)

print("""
=== MLFLOW - Built into Databricks ===

MLflow is pre-configured in Databricks - use it!

# 1. EXPERIMENT TRACKING
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Start experiment
mlflow.set_experiment("/Users/you@company.com/churn_prediction")

with mlflow.start_run(run_name="rf_baseline"):
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Log artifacts (plots, data samples, etc.)
    mlflow.log_artifact("confusion_matrix.png")


# 2. MODEL REGISTRY
# Register model for deployment
mlflow.register_model(
    "runs:/abc123/model",
    "churn_prediction_model"
)

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="churn_prediction_model",
    version=1,
    stage="Production"
)


# 3. MODEL SERVING
# Load production model
model = mlflow.pyfunc.load_model("models:/churn_prediction_model/Production")

# Batch inference with Spark
predictions = model.predict(spark_df.toPandas())

# Or use Spark UDF for distributed inference
predict_udf = mlflow.pyfunc.spark_udf(spark, "models:/churn_prediction_model/Production")
predictions_df = spark_df.withColumn("prediction", predict_udf(*feature_columns))


# 4. FEATURE STORE (Databricks)
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

# Create feature table
fs.create_table(
    name="ml.features.user_features",
    primary_keys=["user_id"],
    df=user_features_df,
    description="User features for ML models"
)

# Read features for training
training_set = fs.create_training_set(
    df=labels_df,
    feature_lookups=[
        FeatureLookup(
            table_name="ml.features.user_features",
            lookup_key="user_id"
        )
    ],
    label="churned"
)
""")


# =============================================================================
# SECTION 7: Job Scheduling and Workflows
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 7: Job Scheduling and Workflows")
print("=" * 70)

print("""
=== DATABRICKS WORKFLOWS ===

Your company's Databricks has built-in job scheduling:

1. NOTEBOOK JOBS
   - Schedule any notebook to run
   - Parameterize with widgets
   - Email alerts on failure

2. PIPELINE ORCHESTRATION
   - Chain notebooks together
   - Conditional branching
   - Parallel execution

Example notebook structure for ETL pipeline:

/Repos/team/etl-pipeline/
├── 01_bronze_ingestion.py    # Read raw data
├── 02_silver_cleaning.py     # Clean and validate
├── 03_gold_aggregation.py    # Business metrics
├── 04_data_quality_checks.py # Validation
└── config/
    └── parameters.py         # Shared configuration


# In 01_bronze_ingestion.py:
dbutils.widgets.text("date", "")
date = dbutils.widgets.get("date")

df = spark.read.json(f"/mnt/raw/events/date={date}/")
df.write.format("delta").mode("append").save("/mnt/bronze/events")

dbutils.notebook.exit(f"Processed {df.count()} records")


# In orchestration notebook:
date = "2024-01-15"

result1 = dbutils.notebook.run("01_bronze_ingestion", 3600, {"date": date})
print(f"Bronze: {result1}")

result2 = dbutils.notebook.run("02_silver_cleaning", 3600, {"date": date})
print(f"Silver: {result2}")

result3 = dbutils.notebook.run("03_gold_aggregation", 3600, {"date": date})
print(f"Gold: {result3}")


=== DELTA LIVE TABLES (DLT) ===

Declarative ETL - define WHAT, not HOW:

import dlt

@dlt.table
def bronze_events():
    return spark.read.json("/mnt/raw/events/")

@dlt.table
@dlt.expect_or_drop("valid_amount", "amount >= 0")
def silver_events():
    return dlt.read("bronze_events") \\
        .filter(F.col("event_type").isNotNull()) \\
        .dropDuplicates(["event_id"])

@dlt.table
def gold_daily_metrics():
    return dlt.read("silver_events") \\
        .groupBy(F.to_date("event_time").alias("date")) \\
        .agg(
            F.count("*").alias("total_events"),
            F.sum("amount").alias("total_revenue")
        )
""")


# =============================================================================
# SECTION 8: Real-World Patterns
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 8: Real-World Patterns You'll Use Daily")
print("=" * 70)

# --- 8.1 User Session Analysis ---
print("\n--- Pattern: Sessionization ---")

window_spec = Window.partitionBy("user_id").orderBy("event_time")

sessionized = (
    events_df
    .withColumn("prev_time", F.lag("event_time").over(window_spec))
    .withColumn("time_diff_minutes",
        (F.unix_timestamp("event_time") - F.unix_timestamp("prev_time")) / 60
    )
    .withColumn("is_new_session",
        F.when(
            F.col("prev_time").isNull() | (F.col("time_diff_minutes") > 30),
            1
        ).otherwise(0)
    )
    .withColumn("session_id",
        F.concat(
            F.col("user_id"),
            F.lit("_"),
            F.sum("is_new_session").over(window_spec)
        )
    )
)

print("Sessionized events:")
sessionized.select("user_id", "event_time", "event_type", "session_id").show()


# --- 8.2 Funnel Analysis ---
print("\n--- Pattern: Conversion Funnel ---")

funnel_df = (
    events_df
    .groupBy("user_id")
    .agg(
        F.max(F.when(F.col("event_type") == "page_view", 1).otherwise(0)).alias("viewed"),
        F.max(F.when(F.col("event_type") == "add_to_cart", 1).otherwise(0)).alias("added_to_cart"),
        F.max(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("purchased")
    )
)

funnel_summary = funnel_df.agg(
    F.count("*").alias("total_users"),
    F.sum("viewed").alias("viewed"),
    F.sum("added_to_cart").alias("added_to_cart"),
    F.sum("purchased").alias("purchased")
)

print("Funnel Summary:")
funnel_summary.show()


# --- 8.3 Cohort Retention ---
print("\n--- Pattern: Cohort Analysis ---")

# Would use real user data in production
user_activity = spark.createDataFrame([
    (1, "2024-01-01", "2024-01-01"),
    (1, "2024-01-01", "2024-01-15"),
    (1, "2024-01-01", "2024-02-01"),
    (2, "2024-01-15", "2024-01-15"),
    (2, "2024-01-15", "2024-01-20"),
    (3, "2024-02-01", "2024-02-01"),
    (3, "2024-02-01", "2024-02-15"),
    (3, "2024-02-01", "2024-03-01"),
], ["user_id", "signup_date", "activity_date"])

user_activity = user_activity \
    .withColumn("signup_date", F.to_date("signup_date")) \
    .withColumn("activity_date", F.to_date("activity_date"))

cohort_analysis = (
    user_activity
    .withColumn("cohort_month", F.date_trunc("month", "signup_date"))
    .withColumn("activity_month", F.date_trunc("month", "activity_date"))
    .withColumn("months_since_signup",
        F.months_between("activity_month", "cohort_month").cast("int")
    )
    .groupBy("cohort_month", "months_since_signup")
    .agg(F.countDistinct("user_id").alias("active_users"))
    .orderBy("cohort_month", "months_since_signup")
)

print("Cohort Retention:")
cohort_analysis.show()


# =============================================================================
# SECTION 9: Interview Tips - Leverage Your Infrastructure
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 9: Interview Tips - Stand Out")
print("=" * 70)

print("""
=== HOW TO LEVERAGE COMPANY INFRASTRUCTURE IN INTERVIEWS ===

When asked "Tell me about a project you've worked on":

GOOD Answer (Generic):
"I processed log data using PySpark."

GREAT Answer (Shows Infrastructure Knowledge):
"I built an incremental ETL pipeline on Databricks that processes
500GB of event data daily. I used Delta Lake for ACID transactions,
implemented the medallion architecture (bronze/silver/gold), and
set up Autoloader for real-time ingestion. The pipeline includes
data quality checks with Delta Live Tables expectations, and we
track all transformations with Unity Catalog lineage."

=== PHRASES THAT IMPRESS ===

Instead of...              Say...
─────────────────────────────────────────────────────────
"I read parquet files"     "I leveraged Delta Lake's time travel
                            for data recovery and auditing"

"I scheduled jobs"         "I orchestrated multi-notebook workflows
                            with dependency management and alerting"

"I joined tables"          "I optimized joins using broadcast hints
                            and Z-ordering for frequently filtered columns"

"I trained a model"        "I tracked experiments in MLflow, registered
                            models to the Model Registry, and deployed
                            with batch inference using Spark UDFs"

"I cleaned data"           "I implemented the medallion architecture with
                            schema evolution and data quality expectations"


=== QUESTIONS TO ASK IN INTERVIEWS ===

1. "What's your current data lake architecture? Do you use
    Delta Lake or Iceberg?"

2. "How do you handle schema evolution in your pipelines?"

3. "Do you use Unity Catalog for data governance?"

4. "How do you track ML experiments - MLflow or another tool?"

5. "What's your approach to incremental vs full refresh for
    large datasets?"

These questions show you understand enterprise data engineering!
""")


# =============================================================================
# SECTION 10: Cheat Sheet
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 10: Quick Reference Cheat Sheet")
print("=" * 70)

print("""
=== PYSPARK + DATABRICKS CHEAT SHEET ===

# READ DATA
spark.read.format("delta").load("/path")
spark.table("catalog.schema.table")
spark.read.parquet("s3://bucket/path")

# WRITE DATA
df.write.format("delta").mode("overwrite").partitionBy("date").save("/path")
df.write.saveAsTable("catalog.schema.table")

# DELTA OPERATIONS
OPTIMIZE delta.`/path` ZORDER BY (col1, col2)
VACUUM delta.`/path` RETAIN 168 HOURS
DESCRIBE HISTORY delta.`/path`

# TIME TRAVEL
spark.read.format("delta").option("versionAsOf", 5).load("/path")
spark.read.format("delta").option("timestampAsOf", "2024-01-15").load("/path")

# MERGE (Upsert)
delta_table.alias("t").merge(
    updates.alias("s"),
    "t.id = s.id"
).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

# WINDOW FUNCTIONS
window = Window.partitionBy("user").orderBy("date")
df.withColumn("running_total", F.sum("amount").over(window))
df.withColumn("prev_value", F.lag("amount", 1).over(window))
df.withColumn("rank", F.row_number().over(window))

# PERFORMANCE
df.cache()  # Cache in memory
F.broadcast(small_df)  # Broadcast join
df.repartition("key")  # Repartition by column
df.coalesce(10)  # Reduce partitions

# DATABRICKS UTILITIES
dbutils.fs.ls("/path")
dbutils.secrets.get("scope", "key")
dbutils.widgets.text("param", "default")
dbutils.notebook.run("/notebook", timeout, {"param": "value"})

# MLFLOW
mlflow.start_run()
mlflow.log_param("key", value)
mlflow.log_metric("accuracy", 0.95)
mlflow.sklearn.log_model(model, "model")
mlflow.register_model("runs:/id/model", "model_name")
""")


# Cleanup
print("\n" + "=" * 70)
print("TUTORIAL COMPLETE!")
print("=" * 70)

print("""
KEY TAKEAWAYS:

1. Your company PAYS for Spark/Databricks - USE IT!
2. Delta Lake gives you ACID, time travel, and MERGE
3. Medallion architecture (Bronze/Silver/Gold) is the standard
4. MLflow is FREE and built-in - track everything
5. Photon + AQE = automatic performance gains
6. Know these patterns for interviews

Next steps:
- Practice these patterns in your company's Databricks workspace
- Build a portfolio project using these techniques
- Mention specific features (Delta Lake, MLflow, etc.) in interviews
""")

# spark.stop()  # Uncomment in production
