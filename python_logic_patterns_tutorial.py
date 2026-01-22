"""
=============================================================================
Python Logic Patterns for Data Engineers - "Genius" Tricks
=============================================================================
This tutorial covers clever uses of sets, comprehensions, and logic patterns
that make data engineering code elegant, efficient, and production-ready.

These patterns are:
- Commonly used in ETL pipelines
- Asked about in interviews
- Essential for data validation
- Much faster than naive approaches
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import groupby, chain
from functools import reduce
from typing import List, Dict, Set, Any, Optional
from datetime import datetime, timedelta

# =============================================================================
# SECTION 1: Set Operations - The Foundation
# =============================================================================

print("=" * 70)
print("SECTION 1: Set Operations")
print("=" * 70)

# --- 1.1 Finding Missing/Extra Elements ---

# The pattern you highlighted - VERY common in data validation
required_columns = {'user_id', 'timestamp', 'event_type', 'value'}
actual_columns = {'user_id', 'timestamp', 'event_type', 'extra_col'}

missing = required_columns - actual_columns  # {'value'}
extra = actual_columns - required_columns    # {'extra_col'}
common = required_columns & actual_columns   # intersection
all_cols = required_columns | actual_columns # union

print(f"Required: {required_columns}")
print(f"Actual: {actual_columns}")
print(f"Missing: {missing}")
print(f"Extra: {extra}")
print(f"Common: {common}")
print(f"All: {all_cols}")

# PRODUCTION EXAMPLE: Schema validation
def validate_schema(df: pd.DataFrame, required: set, optional: set = None) -> dict:
    """
    Validate DataFrame schema against requirements.
    Returns detailed validation report.
    """
    optional = optional or set()
    actual = set(df.columns)

    return {
        'valid': required <= actual,  # All required present (subset check)
        'missing_required': required - actual,
        'missing_optional': optional - actual,
        'unexpected': actual - (required | optional),
        'present': actual & (required | optional)
    }

# Test it
df = pd.DataFrame({'user_id': [1], 'timestamp': ['2024-01-01'], 'extra': [1]})
result = validate_schema(df, required={'user_id', 'timestamp', 'value'}, optional={'metadata'})
print(f"\nSchema Validation: {result}")


# --- 1.2 Symmetric Difference - What's Different? ---

# Find elements in either set but NOT both
old_users = {'alice', 'bob', 'charlie', 'diana'}
new_users = {'bob', 'charlie', 'eve', 'frank'}

changed = old_users ^ new_users  # symmetric difference
print(f"\nUsers that changed (added or removed): {changed}")

# PRODUCTION EXAMPLE: Detect schema drift
def detect_schema_drift(old_schema: set, new_schema: set) -> dict:
    """Detect changes between two schema versions."""
    return {
        'added_columns': new_schema - old_schema,
        'removed_columns': old_schema - new_schema,
        'unchanged_columns': old_schema & new_schema,
        'has_drift': old_schema != new_schema
    }

drift = detect_schema_drift(
    {'id', 'name', 'email', 'created_at'},
    {'id', 'name', 'email', 'updated_at', 'status'}
)
print(f"Schema Drift: {drift}")


# --- 1.3 Subset/Superset Checks ---

permissions_required = {'read', 'write'}
user_permissions = {'read', 'write', 'delete', 'admin'}

# Check if user has all required permissions
has_access = permissions_required <= user_permissions  # subset check
print(f"\nUser has required permissions: {has_access}")

# PRODUCTION EXAMPLE: Feature flag validation
def validate_feature_dependencies(
    enabled_features: set,
    feature_dependencies: dict
) -> dict:
    """
    Check if all dependencies are satisfied for enabled features.
    feature_dependencies: {feature: {required_features}}
    """
    issues = {}
    for feature in enabled_features:
        required = feature_dependencies.get(feature, set())
        missing = required - enabled_features
        if missing:
            issues[feature] = missing
    return issues

deps = {
    'advanced_analytics': {'basic_analytics', 'data_export'},
    'ml_predictions': {'advanced_analytics', 'api_access'},
    'real_time_dashboard': {'basic_analytics'}
}
enabled = {'basic_analytics', 'ml_predictions', 'api_access'}

issues = validate_feature_dependencies(enabled, deps)
print(f"Feature dependency issues: {issues}")


# =============================================================================
# SECTION 2: Dictionary Tricks
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: Dictionary Patterns")
print("=" * 70)

# --- 2.1 Dictionary Difference/Comparison ---

old_config = {'host': 'localhost', 'port': 5432, 'timeout': 30, 'pool_size': 10}
new_config = {'host': 'prod-db.com', 'port': 5432, 'timeout': 60, 'max_connections': 100}

# Find what changed
changed_keys = {k for k in old_config.keys() & new_config.keys()
                if old_config[k] != new_config[k]}
added_keys = new_config.keys() - old_config.keys()
removed_keys = old_config.keys() - new_config.keys()

print(f"Changed: {changed_keys}")  # {'host', 'timeout'}
print(f"Added: {added_keys}")      # {'max_connections'}
print(f"Removed: {removed_keys}")  # {'pool_size'}

# PRODUCTION EXAMPLE: Config diff for audit logging
def diff_configs(old: dict, new: dict) -> dict:
    """Generate detailed diff between two configs."""
    all_keys = old.keys() | new.keys()

    diff = {
        'added': {k: new[k] for k in new.keys() - old.keys()},
        'removed': {k: old[k] for k in old.keys() - new.keys()},
        'modified': {
            k: {'old': old[k], 'new': new[k]}
            for k in old.keys() & new.keys()
            if old[k] != new[k]
        },
        'unchanged': {k: old[k] for k in old.keys() & new.keys() if old[k] == new[k]}
    }
    return diff

print(f"\nConfig Diff: {diff_configs(old_config, new_config)}")


# --- 2.2 Dictionary Merge with Priority ---

defaults = {'timeout': 30, 'retries': 3, 'log_level': 'INFO'}
env_config = {'timeout': 60, 'debug': True}
cli_args = {'log_level': 'DEBUG'}

# Priority: cli_args > env_config > defaults (later dicts override earlier)
final_config = {**defaults, **env_config, **cli_args}
print(f"\nMerged config: {final_config}")

# Python 3.9+ alternative
# final_config = defaults | env_config | cli_args


# --- 2.3 Inverting a Dictionary ---

# Original: code -> description
status_codes = {200: 'OK', 404: 'Not Found', 500: 'Server Error'}

# Inverted: description -> code
status_names = {v: k for k, v in status_codes.items()}
print(f"\nInverted dict: {status_names}")

# PRODUCTION EXAMPLE: Bidirectional mapping
class BiDict:
    """Bidirectional dictionary for code/name lookups."""
    def __init__(self, mapping: dict):
        self.forward = mapping
        self.reverse = {v: k for k, v in mapping.items()}

    def get_by_code(self, code):
        return self.forward.get(code)

    def get_by_name(self, name):
        return self.reverse.get(name)

event_types = BiDict({1: 'click', 2: 'view', 3: 'purchase'})
print(f"Code 1 -> {event_types.get_by_code(1)}")
print(f"'purchase' -> {event_types.get_by_name('purchase')}")


# --- 2.4 Grouping with defaultdict ---

events = [
    {'user': 'alice', 'event': 'click'},
    {'user': 'bob', 'event': 'view'},
    {'user': 'alice', 'event': 'purchase'},
    {'user': 'alice', 'event': 'click'},
    {'user': 'bob', 'event': 'click'},
]

# Group events by user
by_user = defaultdict(list)
for e in events:
    by_user[e['user']].append(e['event'])

print(f"\nGrouped by user: {dict(by_user)}")

# Count events per user
event_counts = defaultdict(lambda: defaultdict(int))
for e in events:
    event_counts[e['user']][e['event']] += 1

print(f"Event counts: {dict(event_counts)}")


# =============================================================================
# SECTION 3: List/Comprehension Patterns
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: List/Comprehension Patterns")
print("=" * 70)

# --- 3.1 Flatten Nested Lists ---

nested = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]

# Method 1: List comprehension
flat = [item for sublist in nested for item in sublist]
print(f"Flattened: {flat}")

# Method 2: chain (better for large data)
from itertools import chain
flat = list(chain.from_iterable(nested))

# PRODUCTION EXAMPLE: Flatten nested JSON responses
api_responses = [
    {'users': [{'id': 1}, {'id': 2}]},
    {'users': [{'id': 3}]},
    {'users': [{'id': 4}, {'id': 5}]}
]

all_users = [user for response in api_responses for user in response['users']]
print(f"All users: {all_users}")


# --- 3.2 Zip Patterns ---

# Transpose rows to columns
rows = [
    ['Alice', 25, 'NYC'],
    ['Bob', 30, 'LA'],
    ['Charlie', 35, 'Chicago']
]

# Transpose using zip
cols = list(zip(*rows))
print(f"\nTransposed: {cols}")

# Create dict from two lists
keys = ['name', 'age', 'city']
values = ['Diana', 28, 'Boston']
record = dict(zip(keys, values))
print(f"Zipped dict: {record}")

# PRODUCTION EXAMPLE: Batch processing with indices
def process_in_batches(items: list, batch_size: int):
    """Yield items in batches."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

data = list(range(10))
for batch_num, batch in enumerate(process_in_batches(data, 3)):
    print(f"Batch {batch_num}: {batch}")


# --- 3.3 Dictionary from Pairs ---

# Two parallel lists to dict
ids = [1, 2, 3, 4]
names = ['Alice', 'Bob', 'Charlie', 'Diana']
id_to_name = dict(zip(ids, names))
print(f"\nID to Name: {id_to_name}")

# With enumeration (index as key)
name_by_index = dict(enumerate(names))
print(f"By Index: {name_by_index}")

# PRODUCTION EXAMPLE: Create lookup table
def create_lookup(df: pd.DataFrame, key_col: str, value_col: str) -> dict:
    """Create efficient lookup dict from DataFrame."""
    return dict(zip(df[key_col], df[value_col]))

users_df = pd.DataFrame({
    'user_id': [1, 2, 3],
    'username': ['alice', 'bob', 'charlie']
})
lookup = create_lookup(users_df, 'user_id', 'username')
print(f"User lookup: {lookup}")


# =============================================================================
# SECTION 4: Boolean Logic Patterns
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Boolean Logic Patterns")
print("=" * 70)

# --- 4.1 Any/All Patterns ---

values = [1, 2, 3, 4, 5]

# Check conditions across collection
all_positive = all(v > 0 for v in values)
any_even = any(v % 2 == 0 for v in values)
none_negative = not any(v < 0 for v in values)

print(f"All positive: {all_positive}")
print(f"Any even: {any_even}")
print(f"None negative: {none_negative}")

# PRODUCTION EXAMPLE: Data quality checks
def validate_data_quality(df: pd.DataFrame) -> dict:
    """Run multiple data quality checks."""
    checks = {
        'no_nulls_in_id': df['user_id'].notna().all(),
        'positive_amounts': (df['amount'] > 0).all() if 'amount' in df.columns else True,
        'valid_dates': pd.to_datetime(df['date'], errors='coerce').notna().all(),
        'no_duplicates': df['user_id'].is_unique,
    }

    return {
        'all_passed': all(checks.values()),
        'checks': checks,
        'failed': [k for k, v in checks.items() if not v]
    }

test_df = pd.DataFrame({
    'user_id': [1, 2, 3],
    'amount': [100, 200, -50],  # Invalid!
    'date': ['2024-01-01', '2024-01-02', '2024-01-03']
})
quality = validate_data_quality(test_df)
print(f"\nData Quality: {quality}")


# --- 4.2 Short-Circuit Evaluation ---

def expensive_check():
    print("Running expensive check...")
    return True

# This won't run expensive_check if first condition is False
result = False and expensive_check()  # expensive_check never called

# PRODUCTION EXAMPLE: Lazy validation chain
def validate_record(record: dict) -> tuple:
    """Validate record with short-circuit evaluation."""
    validations = [
        (lambda r: 'id' in r, "Missing 'id' field"),
        (lambda r: isinstance(r.get('id'), int), "'id' must be integer"),
        (lambda r: r.get('id', 0) > 0, "'id' must be positive"),
        (lambda r: 'name' in r, "Missing 'name' field"),
        (lambda r: len(r.get('name', '')) > 0, "'name' cannot be empty"),
    ]

    for check, error_msg in validations:
        if not check(record):
            return (False, error_msg)

    return (True, "Valid")

print(f"\nValidation: {validate_record({'id': -1, 'name': 'Test'})}")
print(f"Validation: {validate_record({'id': 1, 'name': 'Test'})}")


# --- 4.3 Ternary and Conditional Expressions ---

status = 'active'

# Traditional ternary
label = 'Enabled' if status == 'active' else 'Disabled'

# Chained ternary (use sparingly - can reduce readability)
priority = (
    'critical' if status == 'error' else
    'high' if status == 'warning' else
    'normal' if status == 'active' else
    'low'
)

# PRODUCTION EXAMPLE: Map values with fallback
def categorize_amount(amount: float) -> str:
    """Categorize transaction amount."""
    return (
        'micro' if amount < 10 else
        'small' if amount < 100 else
        'medium' if amount < 1000 else
        'large' if amount < 10000 else
        'enterprise'
    )

amounts = [5, 50, 500, 5000, 50000]
categories = [categorize_amount(a) for a in amounts]
print(f"\nCategories: {list(zip(amounts, categories))}")


# =============================================================================
# SECTION 5: Counter and Frequency Patterns
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Counter Patterns")
print("=" * 70)

from collections import Counter

# --- 5.1 Basic Counting ---

events = ['click', 'view', 'click', 'purchase', 'view', 'click', 'view']
counts = Counter(events)
print(f"Event counts: {counts}")
print(f"Most common: {counts.most_common(2)}")

# --- 5.2 Counter Arithmetic ---

day1_events = Counter({'click': 100, 'view': 200, 'purchase': 10})
day2_events = Counter({'click': 150, 'view': 180, 'purchase': 25})

total = day1_events + day2_events
difference = day2_events - day1_events  # Only positive counts kept

print(f"\nTotal: {total}")
print(f"Increase: {difference}")

# PRODUCTION EXAMPLE: Detect anomalies in event distribution
def detect_distribution_anomaly(
    baseline: Counter,
    current: Counter,
    threshold: float = 0.5
) -> dict:
    """
    Detect if current distribution differs significantly from baseline.
    threshold: max allowed percentage change
    """
    anomalies = {}
    all_keys = baseline.keys() | current.keys()

    for key in all_keys:
        base_val = baseline.get(key, 0)
        curr_val = current.get(key, 0)

        if base_val == 0:
            if curr_val > 0:
                anomalies[key] = {'type': 'new', 'value': curr_val}
        else:
            pct_change = (curr_val - base_val) / base_val
            if abs(pct_change) > threshold:
                anomalies[key] = {
                    'type': 'increase' if pct_change > 0 else 'decrease',
                    'change_pct': round(pct_change * 100, 1)
                }

    return anomalies

baseline = Counter({'click': 100, 'view': 200, 'purchase': 10})
current = Counter({'click': 180, 'view': 190, 'purchase': 5, 'error': 50})

anomalies = detect_distribution_anomaly(baseline, current)
print(f"\nAnomalies detected: {anomalies}")


# --- 5.3 Finding Duplicates ---

data = [1, 2, 3, 2, 4, 3, 5, 2]

# Find duplicates using Counter
counts = Counter(data)
duplicates = {item for item, count in counts.items() if count > 1}
print(f"\nDuplicates: {duplicates}")

# PRODUCTION EXAMPLE: Find duplicate records
def find_duplicate_keys(records: list, key_field: str) -> dict:
    """Find records with duplicate key values."""
    key_counts = Counter(r[key_field] for r in records)
    duplicate_keys = {k for k, v in key_counts.items() if v > 1}

    return {
        'duplicate_keys': duplicate_keys,
        'duplicate_records': [r for r in records if r[key_field] in duplicate_keys]
    }

records = [
    {'id': 1, 'name': 'Alice'},
    {'id': 2, 'name': 'Bob'},
    {'id': 1, 'name': 'Alice Updated'},  # Duplicate!
    {'id': 3, 'name': 'Charlie'}
]
dups = find_duplicate_keys(records, 'id')
print(f"Duplicate analysis: {dups}")


# =============================================================================
# SECTION 6: Data Validation Patterns
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6: Data Validation Patterns")
print("=" * 70)

# --- 6.1 Comprehensive Schema Validator ---

class DataValidator:
    """Production-grade data validator using set operations."""

    def __init__(self, schema: dict):
        """
        schema format:
        {
            'column_name': {
                'type': str/int/float,
                'required': True/False,
                'nullable': True/False,
                'allowed_values': set or None,
                'range': (min, max) or None
            }
        }
        """
        self.schema = schema
        self.required_cols = {
            col for col, spec in schema.items()
            if spec.get('required', False)
        }
        self.optional_cols = {
            col for col, spec in schema.items()
            if not spec.get('required', False)
        }

    def validate(self, df: pd.DataFrame) -> dict:
        """Run all validations and return detailed report."""
        errors = []
        warnings = []

        # Column presence check
        actual_cols = set(df.columns)
        missing_required = self.required_cols - actual_cols
        missing_optional = self.optional_cols - actual_cols
        unexpected = actual_cols - (self.required_cols | self.optional_cols)

        if missing_required:
            errors.append(f"Missing required columns: {missing_required}")
        if missing_optional:
            warnings.append(f"Missing optional columns: {missing_optional}")
        if unexpected:
            warnings.append(f"Unexpected columns: {unexpected}")

        # Per-column validation
        for col, spec in self.schema.items():
            if col not in df.columns:
                continue

            # Null check
            if not spec.get('nullable', True):
                null_count = df[col].isna().sum()
                if null_count > 0:
                    errors.append(f"Column '{col}' has {null_count} null values")

            # Allowed values check
            allowed = spec.get('allowed_values')
            if allowed:
                actual_values = set(df[col].dropna().unique())
                invalid_values = actual_values - allowed
                if invalid_values:
                    errors.append(f"Column '{col}' has invalid values: {invalid_values}")

            # Range check
            value_range = spec.get('range')
            if value_range:
                min_val, max_val = value_range
                out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
                if len(out_of_range) > 0:
                    errors.append(f"Column '{col}' has {len(out_of_range)} out-of-range values")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

# Usage
schema = {
    'user_id': {'type': int, 'required': True, 'nullable': False},
    'status': {'type': str, 'required': True, 'allowed_values': {'active', 'inactive', 'pending'}},
    'age': {'type': int, 'required': False, 'range': (0, 150)},
    'email': {'type': str, 'required': True, 'nullable': False}
}

validator = DataValidator(schema)

test_df = pd.DataFrame({
    'user_id': [1, 2, None],  # Has null!
    'status': ['active', 'inactive', 'deleted'],  # Invalid value!
    'age': [25, 200, 30],  # Out of range!
    'email': ['a@b.com', 'c@d.com', 'e@f.com'],
    'extra_col': [1, 2, 3]  # Unexpected!
})

result = validator.validate(test_df)
print(f"Validation Result:")
print(f"  Valid: {result['valid']}")
print(f"  Errors: {result['errors']}")
print(f"  Warnings: {result['warnings']}")


# --- 6.2 Referential Integrity Check ---

def check_referential_integrity(
    child_df: pd.DataFrame,
    child_key: str,
    parent_df: pd.DataFrame,
    parent_key: str
) -> dict:
    """
    Check if all child keys exist in parent (foreign key validation).
    """
    child_keys = set(child_df[child_key].dropna().unique())
    parent_keys = set(parent_df[parent_key].unique())

    orphans = child_keys - parent_keys

    return {
        'valid': len(orphans) == 0,
        'orphan_keys': orphans,
        'orphan_count': len(orphans),
        'orphan_records': child_df[child_df[child_key].isin(orphans)] if orphans else None
    }

# Example
orders = pd.DataFrame({
    'order_id': [1, 2, 3, 4],
    'user_id': [1, 2, 99, 1]  # user_id 99 doesn't exist!
})

users = pd.DataFrame({
    'user_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

integrity = check_referential_integrity(orders, 'user_id', users, 'user_id')
print(f"\nReferential Integrity: {integrity}")


# =============================================================================
# SECTION 7: ETL Pipeline Patterns
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 7: ETL Pipeline Patterns")
print("=" * 70)

# --- 7.1 Incremental Processing ---

def get_incremental_records(
    new_data: pd.DataFrame,
    existing_keys: set,
    key_column: str
) -> dict:
    """
    Separate new records from updates.
    """
    new_keys = set(new_data[key_column].unique())

    keys_to_insert = new_keys - existing_keys
    keys_to_update = new_keys & existing_keys

    return {
        'inserts': new_data[new_data[key_column].isin(keys_to_insert)],
        'updates': new_data[new_data[key_column].isin(keys_to_update)],
        'insert_count': len(keys_to_insert),
        'update_count': len(keys_to_update)
    }

# Example
existing = {1, 2, 3}
incoming = pd.DataFrame({
    'id': [2, 3, 4, 5],  # 2,3 are updates; 4,5 are inserts
    'value': ['updated', 'updated', 'new', 'new']
})

result = get_incremental_records(incoming, existing, 'id')
print(f"Inserts ({result['insert_count']}):\n{result['inserts']}")
print(f"\nUpdates ({result['update_count']}):\n{result['updates']}")


# --- 7.2 Data Reconciliation ---

def reconcile_datasets(
    source: pd.DataFrame,
    target: pd.DataFrame,
    key_column: str
) -> dict:
    """
    Compare source and target datasets for reconciliation.
    """
    source_keys = set(source[key_column].unique())
    target_keys = set(target[key_column].unique())

    return {
        'in_source_only': source_keys - target_keys,
        'in_target_only': target_keys - source_keys,
        'in_both': source_keys & target_keys,
        'source_count': len(source_keys),
        'target_count': len(target_keys),
        'match_rate': len(source_keys & target_keys) / max(len(source_keys), 1)
    }

source_df = pd.DataFrame({'id': [1, 2, 3, 4, 5]})
target_df = pd.DataFrame({'id': [1, 2, 3, 6, 7]})

recon = reconcile_datasets(source_df, target_df, 'id')
print(f"\nReconciliation: {recon}")


# --- 7.3 Change Data Capture (CDC) Pattern ---

def detect_changes(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    key_column: str,
    compare_columns: list
) -> dict:
    """
    Detect inserts, updates, and deletes between snapshots.
    """
    old_keys = set(old_df[key_column])
    new_keys = set(new_df[key_column])

    # Key-level changes
    inserted_keys = new_keys - old_keys
    deleted_keys = old_keys - new_keys
    potential_update_keys = old_keys & new_keys

    # Find actual updates (values changed)
    if potential_update_keys:
        old_subset = old_df[old_df[key_column].isin(potential_update_keys)].set_index(key_column)
        new_subset = new_df[new_df[key_column].isin(potential_update_keys)].set_index(key_column)

        # Compare values
        updated_keys = set()
        for key in potential_update_keys:
            old_vals = tuple(old_subset.loc[key, compare_columns].values)
            new_vals = tuple(new_subset.loc[key, compare_columns].values)
            if old_vals != new_vals:
                updated_keys.add(key)
    else:
        updated_keys = set()

    unchanged_keys = potential_update_keys - updated_keys

    return {
        'inserted': new_df[new_df[key_column].isin(inserted_keys)],
        'deleted': old_df[old_df[key_column].isin(deleted_keys)],
        'updated': new_df[new_df[key_column].isin(updated_keys)],
        'unchanged_count': len(unchanged_keys),
        'summary': {
            'inserts': len(inserted_keys),
            'deletes': len(deleted_keys),
            'updates': len(updated_keys),
            'unchanged': len(unchanged_keys)
        }
    }

old = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'status': ['active', 'active', 'active', 'active']
})

new = pd.DataFrame({
    'id': [1, 2, 3, 5],  # 4 deleted, 5 inserted
    'name': ['Alice', 'Bob Updated', 'Charlie', 'Eve'],  # 2 updated
    'status': ['active', 'inactive', 'active', 'active']
})

changes = detect_changes(old, new, 'id', ['name', 'status'])
print(f"\nCDC Summary: {changes['summary']}")
print(f"Updated records:\n{changes['updated']}")


# =============================================================================
# SECTION 8: Performance Patterns
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 8: Performance Patterns")
print("=" * 70)

# --- 8.1 Set Membership is O(1) vs List O(n) ---

import time

# Create test data
large_list = list(range(100000))
large_set = set(large_list)
search_values = [50000, 99999, 100001]

# List lookup (SLOW)
start = time.time()
for _ in range(1000):
    for val in search_values:
        _ = val in large_list
list_time = time.time() - start

# Set lookup (FAST)
start = time.time()
for _ in range(1000):
    for val in search_values:
        _ = val in large_set
set_time = time.time() - start

print(f"List lookup: {list_time:.4f}s")
print(f"Set lookup: {set_time:.4f}s")
print(f"Set is {list_time/set_time:.0f}x faster")


# --- 8.2 Use Sets for Filtering ---

# SLOW: List-based filtering
def filter_slow(records, allowed_ids_list):
    return [r for r in records if r['id'] in allowed_ids_list]

# FAST: Set-based filtering
def filter_fast(records, allowed_ids_set):
    return [r for r in records if r['id'] in allowed_ids_set]

records = [{'id': i, 'value': f'val_{i}'} for i in range(10000)]
allowed = list(range(0, 10000, 2))  # Even numbers

start = time.time()
result = filter_slow(records, allowed)
slow_time = time.time() - start

allowed_set = set(allowed)
start = time.time()
result = filter_fast(records, allowed_set)
fast_time = time.time() - start

print(f"\nFiltering - List: {slow_time:.4f}s, Set: {fast_time:.4f}s")


# --- 8.3 Precompute Lookups ---

# PRODUCTION PATTERN: Convert DataFrame column to set for fast lookup
def create_fast_filter(df: pd.DataFrame, column: str, values: list) -> pd.DataFrame:
    """Filter DataFrame using set membership (faster for large value lists)."""
    value_set = set(values)  # Convert once
    mask = df[column].isin(value_set)  # pandas optimizes isin with sets
    return df[mask]


# =============================================================================
# SECTION 9: Interview Questions
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 9: Interview Questions")
print("=" * 70)

"""
QUESTION 1: Find Common Elements
Given multiple lists, find elements that appear in ALL lists.
"""
lists = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
common = set(lists[0])
for lst in lists[1:]:
    common &= set(lst)  # Intersection
print(f"Q1 - Common elements: {common}")

# Alternative using reduce
from functools import reduce
common = reduce(lambda a, b: a & b, map(set, lists))
print(f"Q1 - Using reduce: {common}")


"""
QUESTION 2: First Non-Repeating Character
Find the first character in a string that appears only once.
"""
def first_unique_char(s: str) -> str:
    counts = Counter(s)
    for char in s:
        if counts[char] == 1:
            return char
    return None

print(f"Q2 - First unique in 'aabbcdd': {first_unique_char('aabbcdd')}")


"""
QUESTION 3: Two Sum Using Set
Find if any two numbers in array sum to target.
"""
def has_two_sum(nums: list, target: int) -> bool:
    seen = set()
    for num in nums:
        complement = target - num
        if complement in seen:
            return True
        seen.add(num)
    return False

print(f"Q3 - Two sum [1,2,3,4] target=7: {has_two_sum([1,2,3,4], 7)}")


"""
QUESTION 4: Find Missing Number
Array contains n-1 integers from 1 to n. Find the missing one.
"""
def find_missing(nums: list, n: int) -> int:
    full_set = set(range(1, n + 1))
    return (full_set - set(nums)).pop()

print(f"Q4 - Missing in [1,2,4,5] (n=5): {find_missing([1,2,4,5], 5)}")


"""
QUESTION 5: Anagram Groups
Group words that are anagrams of each other.
"""
def group_anagrams(words: list) -> list:
    groups = defaultdict(list)
    for word in words:
        key = tuple(sorted(word))  # Sorted chars as key
        groups[key].append(word)
    return list(groups.values())

words = ['eat', 'tea', 'tan', 'ate', 'nat', 'bat']
print(f"Q5 - Anagram groups: {group_anagrams(words)}")


print("\n" + "=" * 70)
print("TUTORIAL COMPLETE!")
print("=" * 70)
print("""
Key Takeaways:
1. Use SET operations for membership/comparison - O(1) vs O(n)
2. Set difference (-) finds missing elements
3. Set intersection (&) finds common elements
4. Counter is your friend for frequency analysis
5. defaultdict simplifies grouping operations
6. Short-circuit evaluation with 'and'/'or' for efficiency
7. Convert to set BEFORE filtering large datasets
8. Dictionary comprehensions for fast lookups

Practice these patterns - they appear in 90% of data engineering interviews!
""")
