# Python Tutorials for ML/Data Engineers

This repository contains comprehensive Python tutorials designed for junior ML/Data Engineers preparing for job interviews and production work.

---

## Tutorials Overview

| File | Description | Focus |
|------|-------------|-------|
| [python_ml_tutorial.py](python_ml_tutorial.py) | Python fundamentals for data work | NumPy, Pandas, pipelines |
| [polars_production_tutorial.py](polars_production_tutorial.py) | Modern DataFrame library (10-100x faster than pandas) | Lazy evaluation, ETL patterns |
| [sql_production_tutorial.py](sql_production_tutorial.py) | SQL patterns for production databases | Window functions, analytics |
| [python_logic_patterns_tutorial.py](python_logic_patterns_tutorial.py) | Clever Python patterns and tricks | Sets, validation, performance |
| [python_interview_exercises.py](python_interview_exercises.py) | **Interview prep exercises** | Data structures, algorithms, Big O |
| [pyspark_production_tutorial.py](pyspark_production_tutorial.py) | Distributed computing with PySpark | Big data, clusters, window functions |
| [pyspark_databricks_tutorial.py](pyspark_databricks_tutorial.py) | Enterprise Spark with Databricks | Delta Lake, MLflow, production patterns |

---

## 1. Python ML Tutorial

**File:** `python_ml_tutorial.py`

Covers essential Python skills with hands-on exercises.

### Sections

| Section | Topics |
|---------|--------|
| 1 | **Python Fundamentals** - List comprehensions, dictionary operations, lambda functions |
| 2 | **NumPy** - Array creation, broadcasting, vectorized operations, boolean indexing |
| 3 | **Pandas Essentials** - DataFrame operations, filtering, GroupBy, aggregations, missing data |
| 4 | **Data Pipeline Patterns** - Method chaining, apply and transform |
| 5 | **File I/O** - CSV, JSON, Parquet formats |
| 6 | **Error Handling & Logging** - Custom exceptions, logging setup |
| 7 | **Performance Optimization** - Vectorized vs loop operations |
| 8 | **Mini Project** - Complete sales data pipeline |

### Run
```bash
python python_ml_tutorial.py
```

---

## 2. Polars Production Tutorial

**File:** `polars_production_tutorial.py`

Modern DataFrame library written in Rust - increasingly adopted in production ML pipelines.

### Why Polars?
- 10-100x faster than pandas for large datasets
- Lazy evaluation enables query optimization
- Native parallel execution
- Memory efficient (Arrow-based)

### Sections

| Section | Topics |
|---------|--------|
| 1 | Polars fundamentals, syntax differences from pandas |
| 2 | Expressions (the core of Polars), string operations |
| 3 | **Lazy evaluation** - query optimization, streaming for large files |
| 4 | **GroupBy & Window Functions** - ML feature engineering patterns |
| 5 | Joins (inner, left, anti, semi) |
| 6 | **Production Patterns** - Data validation, Feature engineering pipelines, Complete ETL example |
| 7 | **Interview Exercises** - RFM analysis, Sessionization, Data quality reports |

### Run
```bash
python polars_production_tutorial.py
```

---

## 3. SQL Production Tutorial

**File:** `sql_production_tutorial.py`

SQL patterns using SQLite (applicable to PostgreSQL, BigQuery, Snowflake).

### Sections

| Section | Topics |
|---------|--------|
| 1 | SQL fundamentals review |
| 2 | **JOINs** - inner, left, self joins, finding missing data |
| 3 | **Window Functions** - ROW_NUMBER, RANK, LAG/LEAD, running totals |
| 4 | **CTEs & Subqueries** - clean code organization |
| 5 | **Advanced Analytics** - Funnel analysis, Cohort retention, RFM segmentation, YoY comparison |
| 6 | **Performance Optimization** - indexing, query plans, anti-patterns |
| 7 | **Interview Questions** - Second highest salary, consecutive days, gap analysis |
| 8 | **Production Patterns** - Idempotent loads, incremental processing, SCD Type 2 |

### Run
```bash
python sql_production_tutorial.py
```

---

## 4. Python Logic Patterns Tutorial

**File:** `python_logic_patterns_tutorial.py`

Clever Python patterns for Data Engineering - "genius" tricks that make code elegant and efficient.

### Key Patterns

| Section | Pattern | Example Use Case |
|---------|---------|------------------|
| **1. Set Operations** | `required - actual` | Find missing columns |
| | `actual - required` | Find unexpected columns |
| | `a & b` (intersection) | Find common elements |
| | `a <= b` (subset) | Check permissions |
| | `a ^ b` (symmetric diff) | Detect schema drift |
| **2. Dictionary Tricks** | `old.keys() & new.keys()` | Find changed configs |
| | `{**defaults, **override}` | Merge with priority |
| | `{v: k for k,v in d.items()}` | Invert mapping |
| | `defaultdict(list)` | Group by key |
| **3. List Patterns** | `[x for sub in nested for x in sub]` | Flatten nested lists |
| | `dict(zip(keys, values))` | Parallel lists to dict |
| **4. Boolean Logic** | `all(...)` / `any(...)` | Data quality checks |
| | Short-circuit `and`/`or` | Lazy validation |
| **5. Counter** | `Counter(data)` | Frequency analysis |
| | `counter1 - counter2` | Find increases |
| | Duplicate detection | Find repeated keys |
| **6. Validation** | Schema validator class | Production ETL |
| | Referential integrity | Foreign key checks |
| **7. ETL Patterns** | `new_keys - existing_keys` | Incremental loads |
| | CDC (Change Data Capture) | Detect inserts/updates/deletes |
| **8. Performance** | Set membership O(1) | Fast filtering (100-1000x faster) |
| **9. Interview Qs** | Two Sum, Anagrams, etc. | Common interview questions |

### The Core Pattern

```python
# This pattern appears everywhere in data engineering:
missing_columns = set(required_columns) - set(df.columns)
if missing_columns:
    raise DataValidationError(f"Missing columns: {missing_columns}")
```

### Performance Comparison

```python
# SLOW - O(n) for each check
if value in large_list:  # Linear search

# FAST - O(1) lookup
if value in large_set:   # Hash lookup
```

Sets are **100-1000x faster** for membership testing.

### Run
```bash
python python_logic_patterns_tutorial.py
```

---

## 5. Python Interview Exercises

**File:** `python_interview_exercises.py`

Comprehensive interview preparation with 30+ exercises covering the "4 sacred data structures" and classic coding challenges.

### Why This Tutorial?

- Covers the **exact patterns** asked in Data Engineer interviews
- Focus on **Big O complexity** - crucial for interview discussions
- Real-world scenarios: ETL, data cleaning, JOINs without pandas
- Classic algorithms: Two Sum, FizzBuzz, frequency counters

### Sections

| Section | Topics |
|---------|--------|
| 1 | **Data Structures** - Lists (O(n) vs O(1)), Dicts (hash tables), Sets (membership), Tuples (immutability) |
| 2 | **Pythonic Code** - List comprehensions, generators (`yield`), `enumerate/zip`, error handling |
| 3 | **String Manipulation** - `split/join`, `strip`, f-strings, regex patterns |
| 4 | **Classic Interview Exercises** - FizzBuzz, frequency counter, Two Sum O(n), anagrams, palindromes |
| 5 | **Real DE Cases** - Manual JOINs with dicts, GROUP BY without pandas, deduplication, mini ETL pipeline |
| 6 | **Performance Tips** - When to use each data structure, Big O comparison table |

### Key Patterns

```python
# Frequency counter (appears in 80% of interviews)
conteo = {}
for item in data:
    conteo[item] = conteo.get(item, 0) + 1

# Two Sum O(n) - classic interview question
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return (seen[target - num], i)
        seen[num] = i

# Manual JOIN with lookup table (shows you understand JOINs deeply)
lookup = {u['id']: u for u in users}
result = [{**order, **lookup[order['user_id']]}
          for order in orders if order['user_id'] in lookup]
```

### Big O Complexity Reference

| Operation | List | Dict | Set |
|-----------|------|------|-----|
| Access by index | O(1) | - | - |
| Search | O(n) | O(1) | O(1) |
| Insert | O(n)* | O(1) | O(1) |
| Delete | O(n) | O(1) | O(1) |

*O(1) for append at end

### Run
```bash
python python_interview_exercises.py
```

---

## 6. PySpark Production Tutorial

**File:** `pyspark_production_tutorial.py`

Distributed computing with Apache Spark - industry standard for big data pipelines.

### Why PySpark?
- Process terabytes/petabytes of data
- Distributed computing across clusters
- Native integration with cloud platforms (AWS EMR, Databricks, GCP Dataproc)
- SQL + DataFrame + ML unified API

### Sections

| Section | Topics |
|---------|--------|
| 1 | **SparkSession** - Configuration, DataFrame creation, explicit schemas |
| 2 | **Data I/O** - Reading/writing CSV, Parquet, Delta Lake |
| 3 | **Transformations** - withColumn, filter, conditional logic, string operations |
| 4 | **Aggregations** - GroupBy, pivot tables, rollup/cube for OLAP |
| 5 | **Window Functions** - Ranking, running totals, LAG/LEAD, percentiles |
| 6 | **Joins** - Inner, left, right, anti, semi, broadcast joins |
| 7 | **Spark SQL** - SQL queries, CTEs, complex analytics |
| 8 | **Performance** - Caching, partitioning, broadcast, explain plans |
| 9 | **UDFs** - User defined functions, Pandas UDFs, complex types (arrays, maps, structs) |
| 10 | **Production Patterns** - Schema validation, data quality, incremental processing, SCD Type 2 |
| 11 | **Interview Exercises** - Top N per group, sessionization, gap analysis, pivot/unpivot |

### Key Syntax Comparison

| Operation | Pandas | PySpark |
|-----------|--------|---------|
| Filter | `df[df['col'] > 0]` | `df.filter(F.col('col') > 0)` |
| Select | `df[['a', 'b']]` | `df.select('a', 'b')` |
| New column | `df['new'] = df['a'] * 2` | `df.withColumn('new', F.col('a') * 2)` |
| GroupBy | `df.groupby('a')['b'].sum()` | `df.groupBy('a').agg(F.sum('b'))` |
| Rename | `df.rename(columns={'a': 'b'})` | `df.withColumnRenamed('a', 'b')` |

### Run
```bash
pip install pyspark
python pyspark_production_tutorial.py
```

---

## 7. PySpark + Databricks Enterprise Tutorial

**File:** `pyspark_databricks_tutorial.py`

Leverage your company's existing Spark/Databricks infrastructure to stand out.

### Why This Matters
- Your company PAYS for Spark clusters - they want you to use them!
- Knowing Databricks-specific features makes you more valuable
- Enterprise features = production-ready code = promotions

### Sections

| Section | Topics |
|---------|--------|
| 1 | **Leveraging Infrastructure** - What your cluster gives you for free |
| 2 | **Delta Lake** - ACID transactions, time travel, MERGE/upsert |
| 3 | **Databricks Features** - dbutils, display(), %sql magic, Autoloader |
| 4 | **Production Patterns** - Medallion architecture (Bronze/Silver/Gold), incremental processing, SCD Type 2 |
| 5 | **Performance at Scale** - Partitioning, Z-ordering, caching, Photon engine |
| 6 | **ML Integration** - MLflow tracking, Model Registry, Feature Store |
| 7 | **Job Scheduling** - Workflows, notebook orchestration, Delta Live Tables |
| 8 | **Real-World Patterns** - Sessionization, funnel analysis, cohort retention |
| 9 | **Interview Tips** - How to talk about infrastructure in interviews |
| 10 | **Cheat Sheet** - Quick reference for daily use |

### Key Delta Lake Operations

```python
# Time Travel - Query historical data
df = spark.read.format("delta").option("versionAsOf", 5).load("/path")

# MERGE (Upsert) - Update + Insert in one operation
delta_table.alias("t").merge(
    updates.alias("s"), "t.id = s.id"
).whenMatchedUpdateAll().whenNotMatchedInsertAll().execute()

# Optimize with Z-Ordering
OPTIMIZE delta.`/path` ZORDER BY (user_id, date)
```

### Run
```bash
python pyspark_databricks_tutorial.py
```

---

## Choosing the Right Tool: Polars vs PySpark

### Decision Matrix

| Aspect | Polars | PySpark |
|--------|--------|---------|
| **Data Size** | < 100GB (fits in RAM) | > 100GB (distributed) |
| **Infrastructure** | None needed | Requires cluster |
| **Speed (small data)** | Very fast | Slow (overhead) |
| **Speed (big data)** | Limited by RAM | Fast (distributed) |
| **Learning Curve** | Easier | Medium |
| **Cost** | Free | Cluster costs |

### When to Use Each

```
                        Data Size
    ┌─────────────────────────────────────────────────┐
    │   < 1GB        1-100GB         > 100GB          │
    │     │             │               │             │
    │     ▼             ▼               ▼             │
    │  ┌──────┐    ┌─────────┐    ┌──────────┐       │
    │  │Pandas│    │ Polars  │    │ PySpark  │       │
    │  └──────┘    └─────────┘    └──────────┘       │
    └─────────────────────────────────────────────────┘
```

### Recommendation by Situation

| Your Situation | Best Choice |
|----------------|-------------|
| Learning / Portfolio projects | **Polars** |
| Interview preparation | **PySpark** (more asked) |
| Job at startup (< 100GB data) | **Polars** |
| Job at enterprise with Databricks | **PySpark** |
| Need to process 500GB+ daily | **PySpark** |

**Bottom Line:** Learn BOTH - concepts are the same, only syntax differs!

---

## Interview Topics Covered

These tutorials cover common interview topics for ML/Data Engineer positions:

| Topic | Tutorial(s) |
|-------|-------------|
| **Data Structures (Lists, Dicts, Sets)** | **Interview Exercises**, Python Logic |
| **Big O Complexity** | **Interview Exercises** |
| **Classic Algorithms (Two Sum, FizzBuzz)** | **Interview Exercises**, Python Logic |
| **Manual JOINs/GROUP BY** | **Interview Exercises** |
| RFM Segmentation | Polars, SQL, PySpark |
| Cohort Analysis & Retention | SQL, PySpark Databricks |
| Funnel/Conversion Analysis | SQL, PySpark Databricks |
| Window Functions | Polars, SQL, PySpark |
| Sessionization | Polars, SQL, PySpark, PySpark Databricks |
| Data Quality/Validation | Python Logic, Polars, PySpark, Interview Exercises |
| Set Operations | Python Logic, Interview Exercises |
| Performance Optimization | All |
| ETL Patterns | Polars, Python Logic, PySpark, Interview Exercises |
| Error Handling | Python ML, Interview Exercises |
| Generators & Memory Efficiency | Interview Exercises |
| String Manipulation | Interview Exercises |
| Distributed Computing | PySpark, PySpark Databricks |
| Broadcast Joins | PySpark |
| UDFs & Pandas UDFs | PySpark |
| Delta Lake & ACID | PySpark Databricks |
| Medallion Architecture | PySpark Databricks |
| MLflow & Model Registry | PySpark Databricks |
| Time Travel | PySpark Databricks |
| SCD Type 2 | SQL, PySpark Databricks |

---

## Quick Reference

### Set Operations Cheat Sheet

```python
A = {1, 2, 3}
B = {2, 3, 4}

A - B      # {1}       Difference (in A but not B)
B - A      # {4}       Difference (in B but not A)
A & B      # {2, 3}    Intersection (in both)
A | B      # {1,2,3,4} Union (in either)
A ^ B      # {1, 4}    Symmetric difference (in one but not both)
A <= B     # False     Subset check
A >= B     # False     Superset check
```

### Pandas vs Polars Syntax

| Operation | Pandas | Polars |
|-----------|--------|--------|
| Filter | `df[df['col'] > 0]` | `df.filter(pl.col('col') > 0)` |
| Select | `df[['a', 'b']]` | `df.select(['a', 'b'])` |
| New column | `df['new'] = df['a'] * 2` | `df.with_columns((pl.col('a') * 2).alias('new'))` |
| GroupBy | `df.groupby('a')['b'].sum()` | `df.group_by('a').agg(pl.col('b').sum())` |

---

## Getting Started

1. Install dependencies:
```bash
pip install pandas numpy polars pyspark
```

2. Run any tutorial:
```bash
python <tutorial_file>.py
```

3. Complete the exercises marked with `# YOUR CODE HERE:`

4. Check your solutions against the provided ones at the bottom of each file

---

## Key Takeaways

1. **Use SET operations** for membership/comparison - O(1) vs O(n)
2. **Set difference (-)** finds missing elements
3. **Set intersection (&)** finds common elements
4. **Counter** is your friend for frequency analysis
5. **defaultdict** simplifies grouping operations
6. **Short-circuit evaluation** with `and`/`or` for efficiency
7. **Convert to set BEFORE filtering** large datasets
8. **Window functions** are essential for ML feature engineering
9. **Lazy evaluation** (Polars) optimizes complex pipelines
10. **CTEs** make SQL readable and maintainable
11. **Broadcast joins** for small tables in PySpark
12. **Explicit schemas** are mandatory in production Spark jobs
13. **Pandas UDFs** are much faster than regular PySpark UDFs
14. **Delta Lake** gives you ACID, time travel, and MERGE for free
15. **Medallion architecture** (Bronze/Silver/Gold) is the enterprise standard
16. **MLflow** is built into Databricks - track everything!
17. **Know Big O complexity** - interviewers WILL ask about it
18. **`dict.get(key, default)`** avoids KeyError and is interview gold
19. **Generators (`yield`)** for memory-efficient processing of large files
20. **Manual JOINs with dicts** show deep understanding of how JOINs work

---

## Syntax Comparison: Pandas vs Polars vs PySpark

| Operation | Pandas | Polars | PySpark |
|-----------|--------|--------|---------|
| Filter | `df[df['col'] > 0]` | `df.filter(pl.col('col') > 0)` | `df.filter(F.col('col') > 0)` |
| Select | `df[['a', 'b']]` | `df.select(['a', 'b'])` | `df.select('a', 'b')` |
| New column | `df['new'] = df['a'] * 2` | `df.with_columns((pl.col('a') * 2).alias('new'))` | `df.withColumn('new', F.col('a') * 2)` |
| GroupBy | `df.groupby('a')['b'].sum()` | `df.group_by('a').agg(pl.col('b').sum())` | `df.groupBy('a').agg(F.sum('b'))` |
| Rename | `df.rename(columns={'a': 'b'})` | `df.rename({'a': 'b'})` | `df.withColumnRenamed('a', 'b')` |
| Join | `df1.merge(df2, on='key')` | `df1.join(df2, on='key')` | `df1.join(df2, 'key')` |
| Sort | `df.sort_values('col')` | `df.sort('col')` | `df.orderBy('col')` |

---

*Practice these patterns - they appear in 90% of data engineering interviews!*
