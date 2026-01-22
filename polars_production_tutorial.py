"""
=============================================================================
Polars Tutorial for Data/ML Engineers - Production Ready
=============================================================================
Focus: Job interview preparation and production-grade patterns.
Polars is a blazingly fast DataFrame library written in Rust, increasingly
adopted in production ML pipelines due to its performance advantages over pandas.

Why Polars in Production?
- 10-100x faster than pandas for large datasets
- Lazy evaluation enables query optimization
- Native parallel execution
- Memory efficient (Arrow-based)
- No GIL limitations
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# =============================================================================
# SECTION 1: Polars Fundamentals - The Differences from Pandas
# =============================================================================

print("=" * 60)
print("SECTION 1: Polars Fundamentals")
print("=" * 60)

# --- 1.1 DataFrame Creation ---

# From dictionary
df = pl.DataFrame({
    "user_id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "age": [25, 30, 35, 28, 32],
    "salary": [50000.0, 75000.0, 60000.0, 80000.0, 70000.0],
    "department": ["Engineering", "Sales", "Engineering", "Marketing", "Sales"]
})

print("Basic DataFrame:")
print(df)
print(f"\nSchema: {df.schema}")
print(f"Shape: {df.shape}")

# PRODUCTION TIP: Always define schemas explicitly for data validation
schema = {
    "user_id": pl.Int64,
    "name": pl.Utf8,
    "age": pl.Int32,
    "salary": pl.Float64,
    "department": pl.Utf8
}

# --- 1.2 Key Syntax Differences from Pandas ---

# Pandas: df['column'] or df.column
# Polars: pl.col("column") inside expressions

# Pandas: df[df['age'] > 30]
# Polars: df.filter(pl.col("age") > 30)

# Pandas: df['new_col'] = df['col1'] * 2
# Polars: df.with_columns((pl.col("col1") * 2).alias("new_col"))

# Example: Filter and select
result = (
    df
    .filter(pl.col("age") > 25)
    .select(["name", "age", "salary"])
)
print(f"\nFiltered (age > 25):\n{result}")

# EXERCISE 1.1: Convert this pandas-style operation to Polars
# Pandas: df[df['department'] == 'Engineering'][['name', 'salary']]
# YOUR CODE HERE:
# engineering_staff = ...


# =============================================================================
# SECTION 2: Expressions - The Core of Polars
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 2: Expressions")
print("=" * 60)

# --- 2.1 Column Expressions ---

# Polars uses expressions that are optimized before execution
# Multiple operations in a single pass (no intermediate DataFrames)

df = pl.DataFrame({
    "product": ["A", "B", "A", "B", "A", "B"],
    "region": ["North", "North", "South", "South", "North", "South"],
    "sales": [100, 150, 200, 120, 180, 90],
    "cost": [60, 80, 110, 70, 95, 50],
    "quantity": [10, 15, 20, 12, 18, 9]
})

# Multiple column operations in one pass
result = df.select([
    pl.col("product"),
    pl.col("sales"),
    pl.col("cost"),
    (pl.col("sales") - pl.col("cost")).alias("profit"),
    (pl.col("sales") / pl.col("quantity")).alias("price_per_unit"),
    ((pl.col("sales") - pl.col("cost")) / pl.col("sales") * 100).alias("margin_pct")
])
print(f"Computed columns:\n{result}")

# --- 2.2 Expression Contexts ---

# select() - returns only specified columns
# with_columns() - adds/modifies columns, keeps all others
# filter() - row filtering
# group_by().agg() - aggregations

# with_columns example (keeps existing columns)
df_with_profit = df.with_columns([
    (pl.col("sales") - pl.col("cost")).alias("profit"),
    pl.col("sales").mean().alias("avg_sales")  # Broadcasts scalar to all rows
])
print(f"\nWith new columns:\n{df_with_profit}")

# EXERCISE 2.1: Calculate the following metrics in a single select:
# - total_revenue: sales * quantity
# - profit_per_unit: (sales - cost) / quantity
# - is_profitable: profit > 50 (boolean)
# YOUR CODE HERE:
# metrics = df.select([...])


# --- 2.3 String Operations ---

df_text = pl.DataFrame({
    "raw_name": ["  ALICE SMITH  ", "bob jones", "  Charlie Brown"],
    "email": ["alice@company.com", "bob@company.com", "charlie@company.com"],
    "log_entry": ["ERROR: Connection failed", "INFO: User logged in", "WARN: High latency"]
})

# String processing (common in ETL)
cleaned = df_text.with_columns([
    pl.col("raw_name").str.strip_chars().str.to_titlecase().alias("name"),
    pl.col("email").str.split("@").list.get(0).alias("username"),
    pl.col("log_entry").str.extract(r"^(\w+):").alias("log_level"),
    pl.col("log_entry").str.contains("ERROR").alias("is_error")
])
print(f"\nString operations:\n{cleaned}")

# EXERCISE 2.2: Parse this log data
log_df = pl.DataFrame({
    "log": [
        "2024-01-15 10:23:45 | ERROR | user_id=123 | Failed login attempt",
        "2024-01-15 10:24:01 | INFO | user_id=456 | Successful purchase",
        "2024-01-15 10:24:15 | WARN | user_id=123 | Session timeout"
    ]
})
# Extract: timestamp, level, user_id, message
# YOUR CODE HERE:
# parsed_logs = ...


# =============================================================================
# SECTION 3: Lazy Evaluation - Production Performance
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 3: Lazy Evaluation")
print("=" * 60)

# --- 3.1 Lazy vs Eager ---

# PRODUCTION PATTERN: Always use lazy mode for complex pipelines
# Benefits:
# - Query optimization (predicate pushdown, projection pushdown)
# - Parallel execution planning
# - Memory efficiency

# Create lazy frame
lazy_df = pl.LazyFrame({
    "id": range(1000),
    "value": np.random.randn(1000),
    "category": ["A", "B", "C", "D"] * 250
})

# Build pipeline (nothing executes yet)
pipeline = (
    lazy_df
    .filter(pl.col("value") > 0)
    .with_columns([
        (pl.col("value") * 100).alias("scaled_value"),
        pl.col("category").cast(pl.Categorical)
    ])
    .group_by("category")
    .agg([
        pl.col("scaled_value").mean().alias("avg_value"),
        pl.col("scaled_value").std().alias("std_value"),
        pl.len().alias("count")
    ])
    .sort("avg_value", descending=True)
)

# View the optimized query plan
print("Query Plan:")
print(pipeline.explain())

# Execute
result = pipeline.collect()
print(f"\nResult:\n{result}")

# --- 3.2 Streaming Mode for Large Files ---

# PRODUCTION PATTERN: Process files larger than RAM
# lazy_df = pl.scan_csv("huge_file.csv")
# result = lazy_df.filter(...).group_by(...).agg(...).collect(streaming=True)

# --- 3.3 Reading Large Files Efficiently ---

# PRODUCTION TIP: Use scan_* functions for lazy reading
# pl.scan_csv("data.csv")      # Lazy CSV reading
# pl.scan_parquet("data.parquet")  # Lazy Parquet (best for production)
# pl.scan_ndjson("data.jsonl")  # Lazy JSON lines

# Example with schema enforcement and column selection
# production_df = (
#     pl.scan_parquet("events/*.parquet")
#     .select(["user_id", "event_type", "timestamp"])  # Projection pushdown
#     .filter(pl.col("timestamp") > "2024-01-01")      # Predicate pushdown
#     .collect()
# )

# EXERCISE 3.1: Create a lazy pipeline that:
# - Filters values > 0
# - Groups by category
# - Calculates sum, mean, and count
# - Filters groups with count > 100
# YOUR CODE HERE:
# efficient_pipeline = ...


# =============================================================================
# SECTION 4: GroupBy and Aggregations - Interview Favorites
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 4: GroupBy and Aggregations")
print("=" * 60)

# --- 4.1 Basic Aggregations ---

sales_df = pl.DataFrame({
    "date": pl.date_range(datetime(2024, 1, 1), datetime(2024, 3, 31), eager=True),
    "product": ["Widget", "Gadget", "Widget", "Gadget"] * 23,
    "region": ["North", "South", "East", "West"] * 23,
    "revenue": np.random.uniform(100, 1000, 92).round(2),
    "units": np.random.randint(1, 50, 92)
})

# Multiple aggregations
summary = (
    sales_df
    .group_by(["product", "region"])
    .agg([
        pl.col("revenue").sum().alias("total_revenue"),
        pl.col("revenue").mean().alias("avg_revenue"),
        pl.col("revenue").std().alias("std_revenue"),
        pl.col("units").sum().alias("total_units"),
        pl.len().alias("transaction_count"),
        (pl.col("revenue").sum() / pl.col("units").sum()).alias("revenue_per_unit")
    ])
    .sort(["product", "total_revenue"], descending=[False, True])
)
print(f"Sales Summary:\n{summary}")

# --- 4.2 Window Functions ---
# Critical for feature engineering in ML

df = pl.DataFrame({
    "user_id": [1, 1, 1, 2, 2, 2, 3, 3],
    "timestamp": [
        "2024-01-01", "2024-01-02", "2024-01-03",
        "2024-01-01", "2024-01-02", "2024-01-03",
        "2024-01-01", "2024-01-02"
    ],
    "amount": [100, 150, 200, 50, 75, 100, 300, 250]
}).with_columns(pl.col("timestamp").str.to_date())

# Window functions for feature engineering
features = df.with_columns([
    # Running total per user
    pl.col("amount").cum_sum().over("user_id").alias("cumulative_amount"),

    # Rolling average (last 2 transactions)
    pl.col("amount").rolling_mean(window_size=2).over("user_id").alias("rolling_avg_2"),

    # Rank within user
    pl.col("amount").rank(descending=True).over("user_id").alias("amount_rank"),

    # Lag feature (previous transaction)
    pl.col("amount").shift(1).over("user_id").alias("prev_amount"),

    # Percent change
    (pl.col("amount") - pl.col("amount").shift(1)).over("user_id").alias("amount_change"),

    # User statistics (broadcast to all rows)
    pl.col("amount").mean().over("user_id").alias("user_avg_amount"),
    pl.col("amount").max().over("user_id").alias("user_max_amount"),
])
print(f"\nWindow Features:\n{features}")

# EXERCISE 4.1: Create ML features for user behavior prediction:
# - days_since_first_transaction (per user)
# - transaction_count_so_far (cumulative count per user)
# - amount_vs_user_avg (amount / user's average amount)
# - is_above_median (amount > median for that user)
# YOUR CODE HERE:
# ml_features = ...


# --- 4.3 Dynamic GroupBy (Time-based) ---

# PRODUCTION PATTERN: Time-based aggregations
time_df = pl.DataFrame({
    "timestamp": pl.datetime_range(
        datetime(2024, 1, 1),
        datetime(2024, 1, 31, 23),
        interval="1h",
        eager=True
    ),
    "value": np.random.randn(744).cumsum()
})

# Aggregate by day
daily = (
    time_df
    .group_by_dynamic("timestamp", every="1d")
    .agg([
        pl.col("value").mean().alias("daily_avg"),
        pl.col("value").min().alias("daily_min"),
        pl.col("value").max().alias("daily_max"),
        pl.col("value").last().alias("daily_close"),
        pl.col("value").first().alias("daily_open")
    ])
)
print(f"\nDaily Aggregation (first 5 rows):\n{daily.head()}")


# =============================================================================
# SECTION 5: Joins and Data Integration
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 5: Joins")
print("=" * 60)

# --- 5.1 Different Join Types ---

users = pl.DataFrame({
    "user_id": [1, 2, 3, 4],
    "name": ["Alice", "Bob", "Charlie", "Diana"],
    "tier": ["Gold", "Silver", "Gold", "Bronze"]
})

orders = pl.DataFrame({
    "order_id": [101, 102, 103, 104, 105],
    "user_id": [1, 1, 2, 5, 3],  # Note: user_id=5 doesn't exist in users
    "amount": [100, 200, 150, 300, 250]
})

# Inner join (only matching rows)
inner = users.join(orders, on="user_id", how="inner")
print(f"Inner Join:\n{inner}")

# Left join (all from left, matching from right)
left = users.join(orders, on="user_id", how="left")
print(f"\nLeft Join:\n{left}")

# Anti join (rows in left that have NO match in right)
# PRODUCTION USE: Find users who haven't ordered
no_orders = users.join(orders, on="user_id", how="anti")
print(f"\nAnti Join (users without orders):\n{no_orders}")

# Semi join (rows in left that HAVE a match in right, but don't include right columns)
has_orders = users.join(orders, on="user_id", how="semi")
print(f"\nSemi Join (users with orders):\n{has_orders}")

# EXERCISE 5.1: Given these tables, find:
# 1. All orders with customer and product information
# 2. Products that have never been ordered

customers = pl.DataFrame({
    "customer_id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"]
})

products = pl.DataFrame({
    "product_id": [101, 102, 103, 104],
    "product_name": ["Widget", "Gadget", "Gizmo", "Doohickey"],
    "price": [10.0, 20.0, 15.0, 25.0]
})

order_items = pl.DataFrame({
    "order_id": [1, 2, 3, 4],
    "customer_id": [1, 1, 2, 3],
    "product_id": [101, 102, 101, 103]
})
# YOUR CODE HERE:
# orders_with_details = ...
# never_ordered = ...


# =============================================================================
# SECTION 6: Production Data Pipeline Patterns
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 6: Production Pipeline Patterns")
print("=" * 60)

# --- 6.1 Data Validation Pipeline ---

class DataValidator:
    """Production-grade data validation using Polars."""

    def __init__(self, schema: dict):
        self.schema = schema
        self.validation_errors = []

    def validate_schema(self, df: pl.DataFrame) -> bool:
        """Validate DataFrame matches expected schema."""
        for col, expected_type in self.schema.items():
            if col not in df.columns:
                self.validation_errors.append(f"Missing column: {col}")
            elif df[col].dtype != expected_type:
                self.validation_errors.append(
                    f"Column {col}: expected {expected_type}, got {df[col].dtype}"
                )
        return len(self.validation_errors) == 0

    def validate_not_null(self, df: pl.DataFrame, columns: list) -> pl.DataFrame:
        """Check for null values in critical columns."""
        null_counts = df.select([
            pl.col(c).null_count().alias(f"{c}_nulls")
            for c in columns
        ])
        return null_counts

    def validate_range(self, df: pl.DataFrame, column: str,
                       min_val: float = None, max_val: float = None) -> pl.DataFrame:
        """Validate values are within expected range."""
        conditions = []
        if min_val is not None:
            conditions.append(pl.col(column) >= min_val)
        if max_val is not None:
            conditions.append(pl.col(column) <= max_val)

        if conditions:
            combined = conditions[0]
            for c in conditions[1:]:
                combined = combined & c
            invalid = df.filter(~combined)
            return invalid

        return pl.DataFrame()


# Example usage
validator = DataValidator({
    "user_id": pl.Int64,
    "amount": pl.Float64,
    "timestamp": pl.Date
})

# --- 6.2 Feature Engineering Pipeline ---

class FeatureEngineer:
    """Production feature engineering with Polars."""

    def __init__(self):
        self.feature_columns = []

    def add_time_features(self, df: pl.LazyFrame, date_col: str) -> pl.LazyFrame:
        """Extract time-based features from date column."""
        return df.with_columns([
            pl.col(date_col).dt.year().alias(f"{date_col}_year"),
            pl.col(date_col).dt.month().alias(f"{date_col}_month"),
            pl.col(date_col).dt.weekday().alias(f"{date_col}_weekday"),
            pl.col(date_col).dt.day().alias(f"{date_col}_day"),
            (pl.col(date_col).dt.weekday() >= 5).alias(f"{date_col}_is_weekend"),
        ])

    def add_lag_features(self, df: pl.LazyFrame, value_col: str,
                         group_col: str, lags: list) -> pl.LazyFrame:
        """Add lag features for time series."""
        lag_exprs = [
            pl.col(value_col).shift(lag).over(group_col).alias(f"{value_col}_lag_{lag}")
            for lag in lags
        ]
        return df.with_columns(lag_exprs)

    def add_rolling_features(self, df: pl.LazyFrame, value_col: str,
                             group_col: str, windows: list) -> pl.LazyFrame:
        """Add rolling window features."""
        rolling_exprs = []
        for window in windows:
            rolling_exprs.extend([
                pl.col(value_col).rolling_mean(window).over(group_col)
                    .alias(f"{value_col}_rolling_mean_{window}"),
                pl.col(value_col).rolling_std(window).over(group_col)
                    .alias(f"{value_col}_rolling_std_{window}"),
            ])
        return df.with_columns(rolling_exprs)


# --- 6.3 Complete ETL Pipeline Example ---

def create_ml_dataset(
    transactions_path: str,
    users_path: str,
    output_path: str
) -> None:
    """
    Production ETL pipeline for ML feature creation.

    This demonstrates a real-world pattern for creating ML training data.
    """
    # Lazy load data (schema enforcement in production)
    # transactions = pl.scan_parquet(transactions_path)
    # users = pl.scan_parquet(users_path)

    # Simulated data for demonstration
    transactions = pl.LazyFrame({
        "user_id": [1, 1, 1, 2, 2, 3] * 100,
        "timestamp": pl.date_range(
            datetime(2024, 1, 1),
            datetime(2024, 6, 30),
            interval="1d",
            eager=True
        )[:600],
        "amount": np.random.uniform(10, 500, 600).round(2),
        "category": ["food", "transport", "shopping", "food", "bills", "shopping"] * 100
    })

    users = pl.LazyFrame({
        "user_id": [1, 2, 3],
        "signup_date": ["2023-01-15", "2023-06-01", "2023-09-20"],
        "tier": ["gold", "silver", "bronze"]
    }).with_columns(pl.col("signup_date").str.to_date())

    # Build pipeline
    pipeline = (
        transactions
        # Join user data
        .join(users, on="user_id", how="left")

        # Time features
        .with_columns([
            pl.col("timestamp").dt.month().alias("month"),
            pl.col("timestamp").dt.weekday().alias("weekday"),
            (pl.col("timestamp").dt.weekday() >= 5).alias("is_weekend"),
        ])

        # User-level aggregations (window functions)
        .with_columns([
            pl.col("amount").mean().over("user_id").alias("user_avg_amount"),
            pl.col("amount").std().over("user_id").alias("user_std_amount"),
            pl.col("amount").cum_sum().over("user_id").alias("user_cumulative_spend"),
            pl.len().over("user_id").alias("user_total_transactions"),
        ])

        # Lag features
        .with_columns([
            pl.col("amount").shift(1).over("user_id").alias("prev_amount"),
            pl.col("amount").shift(7).over("user_id").alias("amount_7d_ago"),
        ])

        # Derived features
        .with_columns([
            (pl.col("amount") / pl.col("user_avg_amount")).alias("amount_vs_avg"),
            ((pl.col("amount") - pl.col("prev_amount")) / pl.col("prev_amount"))
                .alias("amount_pct_change"),
        ])

        # Filter out nulls from lag features (first rows)
        .filter(pl.col("prev_amount").is_not_null())

        # Select final features
        .select([
            "user_id", "timestamp", "amount", "category", "tier",
            "month", "weekday", "is_weekend",
            "user_avg_amount", "user_std_amount",
            "user_cumulative_spend", "user_total_transactions",
            "prev_amount", "amount_7d_ago",
            "amount_vs_avg", "amount_pct_change"
        ])
    )

    # Execute and save
    result = pipeline.collect()
    print(f"ML Dataset shape: {result.shape}")
    print(f"Columns: {result.columns}")
    print(f"\nSample:\n{result.head()}")

    # In production: result.write_parquet(output_path)

# Run the pipeline
create_ml_dataset("transactions.parquet", "users.parquet", "ml_features.parquet")


# =============================================================================
# SECTION 7: Interview Questions & Exercises
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 7: Interview Exercises")
print("=" * 60)

"""
EXERCISE 7.1: Customer Segmentation Pipeline

Given customer transaction data:
1. Calculate RFM (Recency, Frequency, Monetary) metrics per customer
2. Assign quartile scores (1-4) for each metric
3. Create customer segments based on RFM scores

This is a VERY common interview question for ML/Data Engineer roles.
"""

# Sample data
customer_transactions = pl.DataFrame({
    "customer_id": [1, 1, 1, 2, 2, 3, 3, 3, 3, 4],
    "transaction_date": [
        "2024-01-15", "2024-02-20", "2024-03-10",
        "2024-01-05", "2024-03-25",
        "2024-02-01", "2024-02-15", "2024-03-01", "2024-03-28",
        "2024-01-20"
    ],
    "amount": [100, 150, 200, 500, 300, 50, 75, 60, 80, 1000]
}).with_columns(pl.col("transaction_date").str.to_date())

reference_date = datetime(2024, 4, 1).date()

# YOUR CODE HERE:
# rfm_analysis = ...


"""
EXERCISE 7.2: Sessionization

Given clickstream data, create user sessions:
- A new session starts if there's more than 30 minutes gap between events
- Calculate session-level metrics

This tests window functions and business logic implementation.
"""

clickstream = pl.DataFrame({
    "user_id": [1, 1, 1, 1, 1, 2, 2, 2],
    "event_time": [
        "2024-01-01 10:00:00",
        "2024-01-01 10:05:00",
        "2024-01-01 10:08:00",
        "2024-01-01 11:00:00",  # New session (>30 min gap)
        "2024-01-01 11:02:00",
        "2024-01-01 09:00:00",
        "2024-01-01 09:15:00",
        "2024-01-01 10:00:00",  # New session
    ],
    "page": ["home", "product", "cart", "home", "checkout", "home", "product", "home"]
}).with_columns(pl.col("event_time").str.to_datetime())

# YOUR CODE HERE:
# sessionized = ...


"""
EXERCISE 7.3: Data Quality Report

Create a function that generates a data quality report including:
- Null percentages per column
- Unique value counts
- Numeric column statistics
- Potential data issues (negative values where unexpected, etc.)
"""

def generate_data_quality_report(df: pl.DataFrame) -> pl.DataFrame:
    """Generate comprehensive data quality report."""
    # YOUR CODE HERE:
    pass


# =============================================================================
# SOLUTIONS
# =============================================================================

"""
Scroll down for solutions...
.
.
.
.
.
.
.
.
.
.
"""

# --- SOLUTION 7.1: RFM Analysis ---
print("\n--- Solution 7.1: RFM Analysis ---")

rfm = (
    customer_transactions
    .group_by("customer_id")
    .agg([
        # Recency: days since last transaction
        (pl.lit(reference_date) - pl.col("transaction_date").max())
            .dt.total_days().alias("recency"),
        # Frequency: number of transactions
        pl.len().alias("frequency"),
        # Monetary: total spend
        pl.col("amount").sum().alias("monetary")
    ])
    .with_columns([
        # Quartile scores (4 = best)
        (5 - pl.col("recency").qcut(4, labels=["1", "2", "3", "4"]).cast(pl.Int32))
            .alias("r_score"),
        pl.col("frequency").qcut(4, labels=["1", "2", "3", "4"]).cast(pl.Int32)
            .alias("f_score"),
        pl.col("monetary").qcut(4, labels=["1", "2", "3", "4"]).cast(pl.Int32)
            .alias("m_score"),
    ])
    .with_columns([
        # Combined RFM score
        (pl.col("r_score").cast(pl.Utf8) +
         pl.col("f_score").cast(pl.Utf8) +
         pl.col("m_score").cast(pl.Utf8)).alias("rfm_segment")
    ])
)
print(f"RFM Analysis:\n{rfm}")

# --- SOLUTION 7.2: Sessionization ---
print("\n--- Solution 7.2: Sessionization ---")

sessionized = (
    clickstream
    .sort(["user_id", "event_time"])
    .with_columns([
        # Calculate time difference from previous event
        (pl.col("event_time") - pl.col("event_time").shift(1).over("user_id"))
            .dt.total_minutes().alias("minutes_since_prev"),
    ])
    .with_columns([
        # Mark new session if gap > 30 minutes or first event
        (
            pl.col("minutes_since_prev").is_null() |
            (pl.col("minutes_since_prev") > 30)
        ).alias("is_new_session")
    ])
    .with_columns([
        # Create session ID using cumulative sum of new session flags
        pl.col("is_new_session").cum_sum().over("user_id").alias("session_id")
    ])
)
print(f"Sessionized:\n{sessionized}")

# Session metrics
session_metrics = (
    sessionized
    .group_by(["user_id", "session_id"])
    .agg([
        pl.col("event_time").min().alias("session_start"),
        pl.col("event_time").max().alias("session_end"),
        pl.len().alias("page_views"),
        pl.col("page").n_unique().alias("unique_pages"),
    ])
    .with_columns([
        (pl.col("session_end") - pl.col("session_start"))
            .dt.total_minutes().alias("session_duration_min")
    ])
)
print(f"\nSession Metrics:\n{session_metrics}")


print("\n" + "=" * 60)
print("Tutorial Complete!")
print("=" * 60)
