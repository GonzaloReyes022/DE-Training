"""
=============================================================================
Python Tutorial for Junior ML/Data Engineers
=============================================================================
This tutorial covers essential Python skills with hands-on exercises.
Each section has examples followed by exercises to complete.
"""

# =============================================================================
# SECTION 1: Python Fundamentals for Data Work
# =============================================================================

# --- 1.1 List Comprehensions ---
# Essential for data transformation

# Example: Square numbers
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]
print(f"Squares: {squares}")

# Example: Filter and transform
data = [10, 25, 30, 45, 50]
filtered = [x * 2 for x in data if x > 20]
print(f"Filtered and doubled: {filtered}")

# EXERCISE 1.1: Create a list comprehension that:
# - Takes a list of temperatures in Celsius
# - Converts them to Fahrenheit (F = C * 9/5 + 32)
# - Only includes temperatures above 0°C

celsius = [-10, 0, 15, 22, -5, 30, 100]
# YOUR CODE HERE:
fahrenheit = [x * 9/5 + 32 for x in celsius if x > 0]
print(f"Exercise 1.1 Fahrenheit: {fahrenheit}")
# --- 1.2 Dictionary Operations ---
# Critical for handling JSON data and feature engineering

# Example: Dictionary comprehension
raw_data = {'a': 1, 'b': 2, 'c': 3}
normalized = {k: v / sum(raw_data.values()) for k, v in raw_data.items()}
print(f"Normalized: {normalized}")

# Example: Merging dictionaries
defaults = {'learning_rate': 0.01, 'epochs': 100, 'batch_size': 32}
custom = {'learning_rate': 0.001, 'epochs': 50}
config = {**defaults, **custom}  # custom overwrites defaults
print(f"Config: {config}")

# EXERCISE 1.2: Given a dictionary of student scores, create a new dictionary
# that contains only students who passed (score >= 60) with their letter grades
# A: 90+, B: 80-89, C: 70-79, D: 60-69

scores = {'Alice': 95, 'Bob': 58, 'Charlie': 72, 'Diana': 88, 'Eve': 65}
# YOUR CODE HERE:
grades_A = {k:'A' for k,v in scores.items() if v >= 90}
grades_B = {k:'B' for k,v in scores.items() if 80 <= v < 90}
grades_C = {k:'C' for k,v in scores.items() if 70 <= v < 80}
grades_D = {k:'D' for k,v in scores.items() if 60 <= v < 70}
grades = {**grades_A, **grades_B, **grades_C, **grades_D}
print(f"Exercide 1.2 Grades: {grades}")


# --- 1.3 Lambda Functions and Functional Programming ---
# Common in pandas operations

# Example: Sort by custom key
employees = [
    {'name': 'Alice', 'salary': 50000},
    {'name': 'Bob', 'salary': 75000},
    {'name': 'Charlie', 'salary': 60000}
]
sorted_by_salary = sorted(employees, key=lambda x: x['salary'], reverse=True)
print(f"Sorted: {sorted_by_salary}")

# Example: Map and filter
from functools import reduce

numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
total = reduce(lambda acc, x: acc + x, numbers, 0)
print(f"Doubled: {doubled}, Evens: {evens}, Total: {total}")

# EXERCISE 1.3: Given a list of ML model results, sort them by accuracy descending,
# then by name ascending (for ties)

models = [
    {'name': 'RandomForest', 'accuracy': 0.85},
    {'name': 'XGBoost', 'accuracy': 0.92},
    {'name': 'LogisticRegression', 'accuracy': 0.78},
    {'name': 'SVM', 'accuracy': 0.85},
]
# YOUR CODE HERE:
#convierto a ambos atributos a sortear al mismo tipo, de esta forma cuando hay un empate y va a buscar
#desempatar, lo hace naturalmente con el segundo atributo
sorted_models = sorted(models, key=lambda x:(-x['accuracy'],x['name']))
print(f"Exercise 1.3 Sorted: {sorted_models}")

#otra forma es ordenar primero por nombre y luego por accuaracy, ya que para el segundo sort
#ya entraria los elementos ordenados en caso de empate
#sorted_models = sorted(models,key=lambda x : x['name'])
#sorted_models = sorted(sorted_models, key=lambda x:x['accuracy'], reverse=True)

# =============================================================================
# SECTION 2: Working with NumPy
# =============================================================================

import numpy as np

# --- 2.1 Array Creation and Manipulation ---

# Example: Various ways to create arrays
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
range_arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)  # 5 evenly spaced points between 0 and 1
random_arr = np.random.randn(3, 3)  # Standard normal distribution

print(f"Linspace: {linspace}")
print(f"Random:\n{random_arr}")
matrix = np.zeros((5,5))
matrix = matrix
# EXERCISE 2.1: Create a 5x5 identity matrix, then set all diagonal elements to 2
# YOUR CODE HERE:
matrix = np.eye(5)  # Crea una matriz identidad de 5x5
np.fill_diagonal(matrix, 2)  # Cambia los valores de la diagonal a 2
print(f"Exercise 2.1 Matrix:\n{matrix}")


# --- 2.2 Broadcasting and Vectorized Operations ---

# Example: Broadcasting
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row_means = matrix.mean(axis=1, keepdims=True)
centered = matrix - row_means  # Broadcasting subtracts mean from each row
print(f"Centered matrix:\n{centered}")

# Example: Vectorized operations (avoid loops!)
# BAD (slow):
# result = []
# for x in large_array:
#     result.append(np.sin(x) + np.cos(x))

# GOOD (fast):
large_array = np.random.randn(1000)
result = np.sin(large_array) + np.cos(large_array)

# EXERCISE 2.2: Normalize each column of a matrix to have mean 0 and std 1
# (Z-score normalization: (x - mean) / std)

data = np.array([
    [100, 0.5, 1000],
    [150, 0.8, 1500],
    [120, 0.6, 1200],
    [180, 0.9, 1800]
])
# YOUR CODE HERE:
data_mean = data.mean(axis=0,keepdims=True)
data_std = data.std(axis=0,keepdims=True)
normalized_data = (data - data_mean) / data_std
print(f"Exercise 2.2 Normalized:\n{normalized_data}" )


# --- 2.3 Boolean Indexing and Fancy Indexing ---

# Example: Boolean indexing
arr = np.array([1, -2, 3, -4, 5, -6])
positive = arr[arr > 0]
print(f"Positive elements: {positive}")

# Replace negative values with 0
arr_clipped = arr.copy()
arr_clipped[arr_clipped < 0] = 0
print(f"Clipped: {arr_clipped}")

# EXERCISE 2.3: Given a 2D array of sensor readings, replace all outliers
# (values more than 2 standard deviations from the mean) with the mean value

sensor_data = np.array([
    [22.1, 23.5, 100.0, 22.8],  # 100.0 is an outlier
    [21.9, 22.0, 22.5, 22.1],
    [23.0, -50.0, 22.8, 23.2],  # -50.0 is an outlier
])
# YOUR CODE HERE:
sensor_mean = sensor_data.mean()
sensor_std = sensor_data.std()

cleaned_data = sensor_data.copy()
lower_limit = sensor_mean - 2 * sensor_std
upper_limit = sensor_mean + 2 * sensor_std
outliers_condition = (sensor_data < lower_limit) | (sensor_data > upper_limit)
cleaned_data[outliers_condition] = sensor_mean
 
print(f"Exercise 2.3 Cleaned:\n{cleaned_data}")


# =============================================================================
# SECTION 3: Pandas Essentials
# =============================================================================

import pandas as pd

# --- 3.1 DataFrame Creation and Basic Operations ---

# Example: Create DataFrame from dictionary
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 75000, 60000, 80000],
    'department': ['Engineering', 'Sales', 'Engineering', 'Marketing']
})
print(df)

# Example: Basic operations
print(f"\nDescriptive stats:\n{df.describe()}")
print(f"\nColumn types:\n{df.dtypes}")

# EXERCISE 3.1: Create a DataFrame from the following data and calculate:
# - Mean score per subject
# - Highest scoring student

student_data = {
    'student': ['Ana', 'Ben', 'Cara', 'Dan'],
    'math': [85, 92, 78, 88],
    'science': [90, 85, 92, 79],
    'english': [88, 78, 85, 91]
}
# YOUR CODE HERE:
studen_df = pd.DataFrame(student_data)
df_students = student_data['student']
mean_score_subject = studen_df.drop(columns='student').mean(axis=0)
print(f"Mean score per subject:\n{mean_score_subject}")
highest_scorer = studen_df.drop(columns='student').mean(axis=1)
highest_scorer_name = df_students[highest_scorer.idxmax()]
print(f"Highest scoring student: {highest_scorer_name, highest_scorer.max()}")


# --- 3.2 Data Filtering and Selection ---

# Example: Various selection methods
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=10),
    'value': np.random.randn(10).cumsum(),
    'category': ['A', 'B'] * 5
})

# Filter rows
high_values = df[df['value'] > 0]

# Select columns
subset = df[['date', 'value']]

# loc (label-based) and iloc (integer-based)
first_three = df.iloc[:3]
category_a = df.loc[df['category'] == 'A', ['date', 'value']]

# EXERCISE 3.2: Given a sales DataFrame, find:
# - All sales above $1000 in the 'Electronics' category
# - Total sales per category

sales_df = pd.DataFrame({
    'product': ['Laptop', 'Phone', 'Tablet', 'Headphones', 'TV', 'Speaker'],
    'category': ['Electronics', 'Electronics', 'Electronics', 'Audio', 'Electronics', 'Audio'],
    'price': [1200, 800, 500, 150, 1500, 200],
    'quantity': [5, 20, 15, 50, 3, 30]
})
# YOUR CODE HERE:
high_electronics = sales_df.loc[(sales_df['category'] == 'Electronics') & (sales_df['price'] > 1000), ['product']]
sales_per_category = sales_df.groupby('category')['price'].sum()
print(f"Exercise 3.2 High electronics:\n{high_electronics}")
print(f"Exercise 3.2 Sales per category:\n{sales_per_category}")


# --- 3.3 GroupBy and Aggregations ---

# Example: GroupBy operations
df = pd.DataFrame({
    'region': ['North', 'South', 'North', 'South', 'North', 'South'],
    'product': ['A', 'A', 'B', 'B', 'A', 'B'],
    'sales': [100, 150, 200, 120, 180, 90],
    'returns': [5, 8, 10, 6, 9, 4]
})

# Simple groupby
by_region = df.groupby('region')['sales'].sum()
print(f"Sales by region:\n{by_region}")

# Multiple aggregations
agg_result = df.groupby('region').agg({
    'sales': ['sum', 'mean', 'max'],
    'returns': 'sum'
})
print(f"\nMultiple aggregations:\n{agg_result}")

# EXERCISE 3.3: Given employee data, calculate for each department:
# - Average salary
# - Number of employees
# - Salary range (max - min)

employees_df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'department': ['Engineering', 'Sales', 'Engineering', 'Marketing', 'Sales', 'Engineering'],
    'salary': [75000, 65000, 85000, 70000, 60000, 90000],
    'years_exp': [5, 3, 8, 4, 2, 10]
})
# YOUR CODE HERE:
dept_stats = employees_df.groupby('department').agg({'salary':['mean','count','max','min']})
dept_stats_salary_range = dept_stats['salary']['max'] - dept_stats['salary']['min']
#unir los dos dataframes y agregarlo como una nueva columna, siendo range
dept_department = pd.concat([dept_stats, dept_stats_salary_range], axis=1)
#cambiar nombres de columnas
dept_department.columns = ['mean_salary', 'count', 'max_salary', 'min_salary', 'salary_range']
print(f"Exercise 3.3 Department stats:\n{dept_department}")




# --- 3.4 Handling Missing Data ---

# Example: Missing data operations
df_missing = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': [1, 2, 3, 4, 5]
})

# Check for missing
print(f"Missing values:\n{df_missing.isnull().sum()}")

# Fill missing values
filled_mean = df_missing.fillna(df_missing.mean())
filled_forward = df_missing.fillna(method='ffill')

# Drop rows with missing
dropped = df_missing.dropna()

# EXERCISE 3.4: Clean the following DataFrame:
# - Fill missing 'age' with the median age
# - Fill missing 'city' with 'Unknown'
# - Drop rows where 'income' is missing

messy_df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [25, np.nan, 35, 28, np.nan],
    'city': ['NYC', 'LA', np.nan, 'Chicago', 'NYC'],
    'income': [50000, 60000, np.nan, 70000, 55000]
})
# YOUR CODE HERE:
cleaned_df = messy_df.copy()
cleaned_df['age'] = messy_df['age'].fillna(messy_df['age'].mean())
cleaned_df['city'] = messy_df['city'].fillna('Unknown')
cleaned_df['income'] = messy_df['income'].dropna()
print(f"Exercise 3.4 Cleaned:\n{cleaned_df}")




# =============================================================================
# SECTION 4: Data Pipeline Patterns
# =============================================================================

# --- 4.1 Method Chaining ---

# Example: Clean data pipeline with method chaining
raw_df = pd.DataFrame({
    'Name': ['  alice  ', 'BOB', ' Charlie '],
    'Score': ['85', '90', '78'],
    'Date': ['2024-01-15', '2024-01-16', '2024-01-17']
})

cleaned = (raw_df
    .assign(Name=lambda x: x['Name'].str.strip().str.title())
    .assign(Score=lambda x: pd.to_numeric(x['Score']))
    .assign(Date=lambda x: pd.to_datetime(x['Date']))
    .query('Score > 80')
)
print(f"Cleaned DataFrame:\n{cleaned}")

# EXERCISE 4.1: Create a pipeline that:
# - Reads the data
# - Converts column names to lowercase
# - Filters rows where value > 0
# - Adds a new column 'category' based on value (high if > 50, else low)
# - Sorts by value descending

pipeline_df = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    'Value': [75, -10, 30, 100, 45],
    'Type': ['A', 'B', 'A', 'B', 'A']
})
result = pipeline_df.copy()
result.columns = result.columns.str.lower()
result = result.query('value > 0')
## - Adds a new column 'category' based on value (high if > 50, else low)
result['category'] = ['high' if x > 50 else 'low' for x in result['value']]
result = result.sort_values(by='value', ascending=False)
print(f"Exercise 4.1 Result:\n{result}")

##

# YOUR CODE HERE:
# result = ...


# --- 4.2 Apply and Transform ---

# Example: Custom functions with apply
def categorize_salary(salary):
    if salary < 50000:
        return 'Low'
    elif salary < 80000:
        return 'Medium'
    else:
        return 'High'

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'salary': [45000, 75000, 95000]
})
df['salary_category'] = df['salary'].apply(categorize_salary)
print(df)

# EXERCISE 4.2: Create a function that processes a text column:
# - Converts to lowercase
# - Removes punctuation
# - Counts words
# Apply it to create a 'word_count' column
import string
text_df = pd.DataFrame({
    'id': [1, 2, 3],
    'text': [
        'Hello, World! This is a TEST.',
        'Machine Learning is AMAZING!!!',
        'Data Engineering: Building Pipelines.'
    ]
})
# YOUR CODE HERE:
def process_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

text_df['text'] = text_df['text'].apply(process_text)
text_df['word count'] = text_df['text'].apply(lambda x : len(x.split()))
print(text_df)
    
#     ...
# text_df['word_count'] = ...


# =============================================================================
# SECTION 5: File I/O and Data Formats
# =============================================================================

# --- 5.1 Working with Different File Formats ---

# CSV
# df.to_csv('data.csv', index=False)
# df = pd.read_csv('data.csv')

# JSON
# df.to_json('data.json', orient='records')
# df = pd.read_json('data.json')

# Parquet (efficient columnar format)
# df.to_parquet('data.parquet')
# df = pd.read_parquet('data.parquet')

# Example: Reading with options
# df = pd.read_csv('data.csv',
#     usecols=['col1', 'col2'],  # Only read specific columns
#     dtype={'col1': str},       # Specify data types
#     parse_dates=['date_col'],  # Parse date columns
#     na_values=['NA', 'missing']  # Custom NA values
# )

# EXERCISE 5.1: Write a function that:
# - Takes a DataFrame and a file path
# - Determines the file format from the extension
# - Saves the DataFrame in the appropriate format
# Support: csv, json, parquet

def save_dataframe(df, filepath):
    """Save DataFrame to file based on extension."""
    # YOUR CODE HERE:
    if filepath.endswith('.csv'):
        df.to_csv(filepath, index=False)
    elif filepath.endswith('.json'):
        df.to_json(filepath, orient='records')
    elif filepath.endswith('.parquet'):
        df.to_parquet(filepath)
    return


# =============================================================================
# SECTION 6: Error Handling and Logging
# =============================================================================

import logging

# --- 6.1 Proper Error Handling ---

# Example: Data validation with custom exceptions
class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

def validate_dataframe(df, required_columns, min_rows=1):
    """Validate a DataFrame meets requirements."""
    # Check required columns
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    # Check minimum rows
    if len(df) < min_rows:
        raise DataValidationError(f"DataFrame has {len(df)} rows, minimum is {min_rows}")

    return True

# EXERCISE 6.1: Create a function that safely loads and validates data:
# - Handles FileNotFoundError
# - Validates the data has required columns
# - Returns None and logs error if something fails

def safe_load_csv(filepath, required_columns):
    """Safely load and validate a CSV file."""
    # YOUR CODE HERE:
    try:
        df = pd.read_csv(filepath)
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise DataValidationError(f"Missing columns: {missing_columns}")
        return df
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None




# --- 6.2 Setting Up Logging ---

# Example: Basic logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_data(df):
    """Process data with logging."""
    logger.info(f"Processing DataFrame with {len(df)} rows")

    try:
        result = df.dropna()
        logger.info(f"Dropped {len(df) - len(result)} rows with missing values")
        return result
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise

# EXERCISE 6.2: Add logging to this data pipeline function

def data_pipeline(input_data):
    """Process data through a pipeline."""
    # YOUR CODE HERE: Add appropriate logging statements

    # Step 1: Validate input
    if not isinstance(input_data, pd.DataFrame):
        logger.error(f"Input validation failed: Expected DataFrame, got {type(input_data)}")
        raise TypeError("Input is not a DataFrame")

    # Step 2: Clean data
    try:
        logger.info(f"Cleaning data, dropping na rows...")
        cleaned = input_data.dropna()
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        raise

    # Step 3: Transform
    logger.info(f"Transforming data, scaling...")
    transformed = cleaned.copy()
    for col in cleaned.select_dtypes(include=[np.number]).columns:
        transformed[col] = (cleaned[col] - cleaned[col].mean()) / cleaned[col].std()
    logger.info(f"Data pipeline complete!")
    return transformed


# =============================================================================
# SECTION 7: Performance Optimization
# =============================================================================

# --- 7.1 Efficient Pandas Operations ---

# BAD: Iterating with iterrows (slow)
# for idx, row in df.iterrows():
#     df.loc[idx, 'new_col'] = row['col1'] * 2

# GOOD: Vectorized operations (fast)
# df['new_col'] = df['col1'] * 2

# BETTER for complex logic: apply with axis=1
# df['new_col'] = df.apply(lambda row: complex_function(row), axis=1)

# BEST for very large data: numpy operations
# df['new_col'] = np.where(df['col1'] > 0, df['col1'] * 2, 0)

# EXERCISE 7.1: Optimize this slow function

def slow_processing(df):
    """Slow row-by-row processing - OPTIMIZE THIS."""
    result = df.copy()
    for idx, row in df.iterrows():
        if row['value'] > 0:
            result.loc[idx, 'processed'] = row['value'] * 2 + row['multiplier']
        else:
            result.loc[idx, 'processed'] = 0
    return result

# Test data
test_df = pd.DataFrame({
    'value': np.random.randn(1000),
    'multiplier': np.random.rand(1000) * 10
})

# YOUR OPTIMIZED VERSION:
def fast_processing(df):
    """Optimized vectorized processing."""
    # YOUR CODE HERE:
    test_df = df.copy()
    test_df['processed'] = np.where(test_df['value']>0, test_df['value']*2 + test_df['multiplier'], 0)
    return test_df


# =============================================================================
# SECTION 8: Mini Project - Data Pipeline
# =============================================================================

"""
FINAL PROJECT: Build a complete data processing pipeline

Given messy sales data, create a pipeline that:
1. Loads and validates the data
2. Cleans missing values and outliers
3. Adds derived features (total_revenue, day_of_week, is_weekend)
4. Aggregates data by product and time period
5. Exports summary statistics

Requirements:
- Use proper error handling
- Include logging
- Use efficient pandas operations
- Make the code modular and reusable
"""

# Sample messy data (in real scenario, this would be loaded from a file)
messy_sales = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-02', 'invalid', '2024-01-04', '2024-01-05'] * 20,
    'product': ['Widget', 'Gadget', 'Widget', None, 'Gadget'] * 20,
    'quantity': [10, -5, 15, 8, 1000] * 20,  # Note: negative and outlier values
    'unit_price': [9.99, 19.99, np.nan, 14.99, 19.99] * 20,
    'customer_id': ['C001', 'C002', 'C001', 'C003', 'C002'] * 20
})

# YOUR CODE HERE:
# Build the complete pipeline

class SalesPipeline:
    """Complete sales data processing pipeline."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate(self, df):
        """Validate raw data."""
        # YOUR CODE HERE
        self.logger.info(f"Validating data")
        requiered_colums = ['date', 'product', 'quantity', 'unit_price', 'customer_id']
        missing_columns = set(df.columns) - set(requiered_colums)
        if missing_columns:
            self.logger.error(f"Missing columns")
            raise DataValidationError(f"Missing columns: {missing_columns}")
        #transformar columna date a tipo date
        if type(df['date'].iloc[0]) != 'datetime64[ns]':
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        self.logger.info(f"Data validated")
        return  df

    def clean(self, df):
        """Clean data: handle missing values and outliers."""
        # YOUR CODE HERE
        self.logger.info(f"Cleaning data")
        df = df.dropna()
        df = df[df['quantity'] > 0]
        df = df[df['date'].notnull()]
        df = df[df['product'].notnull()]
        self.logger.info(f"Data cleaned")
        return df


    def add_features(self, df):
        """Add derived features."""
        # YOUR CODE HERE
        self.logger.info(f"Adding features")
        df['total_revenue'] = df['quantity'] * df['unit_price']
        return df


    def aggregate(self, df):
        """Create summary aggregations."""
        # YOUR CODE HERE
        self.logger.info(f"Aggregating data")
        df['sumary'] = df.groupby("product")['total_revenue'].transform('sum')
        return df


    def run(self, df):
        """Run the complete pipeline."""
        # YOUR CODE HERE
        df_final = (df
            .pipe(self.validate)
            .pipe(self.clean)
            .pipe(self.add_features)
            .pipe(self.aggregate)
        )
        return df_final

pipeline = SalesPipeline()
final_sales = pipeline.run(messy_sales)
print(f"Final sales data:\n{final_sales.head()}")

# =============================================================================
# SOLUTIONS (Don't peek until you've tried!)
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

# --- SOLUTIONS ---

# Exercise 1.1
celsius = [-10, 0, 15, 22, -5, 30, 100]
fahrenheit_solution = [c * 9/5 + 32 for c in celsius if c > 0]

# Exercise 1.2
scores = {'Alice': 95, 'Bob': 58, 'Charlie': 72, 'Diana': 88, 'Eve': 65}
def get_grade(score):
    if score >= 90: return 'A'
    elif score >= 80: return 'B'
    elif score >= 70: return 'C'
    else: return 'D'
grades_solution = {name: get_grade(score) for name, score in scores.items() if score >= 60}

# Exercise 1.3
models = [
    {'name': 'RandomForest', 'accuracy': 0.85},
    {'name': 'XGBoost', 'accuracy': 0.92},
    {'name': 'LogisticRegression', 'accuracy': 0.78},
    {'name': 'SVM', 'accuracy': 0.85},
]
sorted_models_solution = sorted(models, key=lambda x: (-x['accuracy'], x['name']))

# Exercise 2.1
matrix_solution = np.eye(5) * 2

# Exercise 2.2
data = np.array([
    [100, 0.5, 1000],
    [150, 0.8, 1500],
    [120, 0.6, 1200],
    [180, 0.9, 1800]
])
normalized_solution = (data - data.mean(axis=0)) / data.std(axis=0)

# Exercise 2.3
sensor_data = np.array([
    [22.1, 23.5, 100.0, 22.8],
    [21.9, 22.0, 22.5, 22.1],
    [23.0, -50.0, 22.8, 23.2],
])
mean_val = sensor_data.mean()
std_val = sensor_data.std()
cleaned_solution = sensor_data.copy()
outlier_mask = np.abs(cleaned_solution - mean_val) > 2 * std_val
cleaned_solution[outlier_mask] = mean_val

print("\n" + "="*50)
print("Tutorial complete! Check your solutions against the provided ones.")
print("="*50)

if __name__ == "__main__":
    pipeline = SalesPipeline()
    final_sales = pipeline.run(messy_sales)
    print(f"Final sales data:\n{final_sales.head()}")
  # The tutorial code is designed to be run as a script for hands-on learning.