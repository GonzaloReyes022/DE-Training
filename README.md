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

## Interview Topics Covered

These tutorials cover common interview topics for ML/Data Engineer positions:

| Topic | Tutorial(s) |
|-------|-------------|
| RFM Segmentation | Polars, SQL |
| Cohort Analysis & Retention | SQL |
| Funnel/Conversion Analysis | SQL |
| Window Functions | Polars, SQL |
| Sessionization | Polars, SQL |
| Data Quality/Validation | Python Logic, Polars |
| Set Operations | Python Logic |
| Performance Optimization | All |
| ETL Patterns | Polars, Python Logic |
| Error Handling | Python ML |

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
pip install pandas numpy polars
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

---

*Practice these patterns - they appear in 90% of data engineering interviews!*
# DE-Training
