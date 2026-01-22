"""
=============================================================================
SQL Tutorial for Data/ML Engineers - Production Ready
=============================================================================
Focus: Job interview preparation and production-grade SQL patterns.
Uses SQLite for local execution, but patterns apply to PostgreSQL,
BigQuery, Snowflake, and other production databases.

Why SQL is Critical:
- Most data lives in databases
- ETL pipelines require SQL expertise
- ML feature stores use SQL queries
- Interview-essential skill
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import random

# =============================================================================
# SETUP: Create SQLite Database with Sample Data
# =============================================================================

def setup_database():
    """Create a production-like database schema with sample data."""
    conn = sqlite3.connect(":memory:")  # In-memory for tutorial
    cursor = conn.cursor()

    # --- Create Tables ---

    # Users table (dimension table)
    cursor.execute("""
        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL,
            tier TEXT CHECK(tier IN ('free', 'basic', 'premium', 'enterprise')),
            signup_date DATE NOT NULL,
            country TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Products table (dimension table)
    cursor.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT NOT NULL,
            price DECIMAL(10, 2) NOT NULL,
            cost DECIMAL(10, 2) NOT NULL,
            is_active BOOLEAN DEFAULT TRUE
        )
    """)

    # Orders table (fact table)
    cursor.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            order_date DATE NOT NULL,
            total_amount DECIMAL(10, 2),
            status TEXT CHECK(status IN ('pending', 'completed', 'cancelled', 'refunded')),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    # Order items table (fact table)
    cursor.execute("""
        CREATE TABLE order_items (
            item_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price DECIMAL(10, 2) NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        )
    """)

    # Events table (for behavioral analytics)
    cursor.execute("""
        CREATE TABLE events (
            event_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            event_type TEXT NOT NULL,
            event_timestamp TIMESTAMP NOT NULL,
            page TEXT,
            session_id TEXT,
            properties TEXT,  -- JSON in production
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    # --- Insert Sample Data ---

    # Users
    users_data = [
        (1, 'alice', 'alice@email.com', 'premium', '2023-01-15', 'USA', True),
        (2, 'bob', 'bob@email.com', 'free', '2023-03-20', 'Canada', True),
        (3, 'charlie', 'charlie@email.com', 'basic', '2023-02-10', 'UK', True),
        (4, 'diana', 'diana@email.com', 'enterprise', '2022-11-05', 'USA', True),
        (5, 'eve', 'eve@email.com', 'premium', '2023-06-01', 'Germany', False),
        (6, 'frank', 'frank@email.com', 'free', '2023-08-15', 'France', True),
        (7, 'grace', 'grace@email.com', 'basic', '2023-04-20', 'USA', True),
        (8, 'henry', 'henry@email.com', 'premium', '2023-05-10', 'Canada', True),
    ]
    cursor.executemany(
        "INSERT INTO users (user_id, username, email, tier, signup_date, country, is_active) VALUES (?, ?, ?, ?, ?, ?, ?)",
        users_data
    )

    # Products
    products_data = [
        (1, 'Widget Pro', 'Electronics', 99.99, 45.00, True),
        (2, 'Gadget Plus', 'Electronics', 149.99, 70.00, True),
        (3, 'Basic Tool', 'Tools', 29.99, 12.00, True),
        (4, 'Premium Kit', 'Kits', 199.99, 95.00, True),
        (5, 'Accessory Pack', 'Accessories', 19.99, 8.00, True),
        (6, 'Discontinued Item', 'Electronics', 79.99, 35.00, False),
    ]
    cursor.executemany(
        "INSERT INTO products (product_id, product_name, category, price, cost, is_active) VALUES (?, ?, ?, ?, ?, ?)",
        products_data
    )

    # Orders and Order Items
    random.seed(42)
    order_id = 1
    for user_id in range(1, 9):
        num_orders = random.randint(1, 5)
        for _ in range(num_orders):
            order_date = f"2024-{random.randint(1,3):02d}-{random.randint(1,28):02d}"
            status = random.choice(['completed', 'completed', 'completed', 'pending', 'cancelled'])

            cursor.execute(
                "INSERT INTO orders (order_id, user_id, order_date, status) VALUES (?, ?, ?, ?)",
                (order_id, user_id, order_date, status)
            )

            # Add items to order
            total = 0
            num_items = random.randint(1, 4)
            for _ in range(num_items):
                product_id = random.randint(1, 5)
                quantity = random.randint(1, 3)
                cursor.execute("SELECT price FROM products WHERE product_id = ?", (product_id,))
                price = cursor.fetchone()[0]
                total += price * quantity

                cursor.execute(
                    "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?)",
                    (order_id, product_id, quantity, price)
                )

            cursor.execute(
                "UPDATE orders SET total_amount = ? WHERE order_id = ?",
                (round(total, 2), order_id)
            )
            order_id += 1

    # Events
    event_types = ['page_view', 'click', 'purchase', 'add_to_cart', 'signup']
    pages = ['home', 'product', 'cart', 'checkout', 'profile']
    event_id = 1
    for user_id in range(1, 9):
        session_id = f"session_{user_id}_{random.randint(1000, 9999)}"
        for _ in range(random.randint(5, 15)):
            event_type = random.choice(event_types)
            page = random.choice(pages)
            timestamp = f"2024-03-{random.randint(1,28):02d} {random.randint(8,22):02d}:{random.randint(0,59):02d}:00"
            cursor.execute(
                "INSERT INTO events (event_id, user_id, event_type, event_timestamp, page, session_id) VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, user_id, event_type, timestamp, page, session_id)
            )
            event_id += 1

    conn.commit()
    return conn


def run_query(conn, query, description=""):
    """Execute query and display results."""
    if description:
        print(f"\n--- {description} ---")
    print(f"Query:\n{query}\n")
    df = pd.read_sql_query(query, conn)
    print(f"Result:\n{df.to_string()}\n")
    return df


# Initialize database
conn = setup_database()

print("=" * 70)
print("SQL TUTORIAL FOR DATA/ML ENGINEERS - PRODUCTION PATTERNS")
print("=" * 70)


# =============================================================================
# SECTION 1: SQL Fundamentals Review
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 1: SQL Fundamentals")
print("=" * 70)

# --- 1.1 SELECT, WHERE, ORDER BY ---

run_query(conn, """
    SELECT
        user_id,
        username,
        tier,
        signup_date,
        country
    FROM users
    WHERE is_active = TRUE
        AND tier IN ('premium', 'enterprise')
    ORDER BY signup_date DESC
""", "Active Premium/Enterprise Users")

# --- 1.2 Aggregations ---

run_query(conn, """
    SELECT
        tier,
        COUNT(*) as user_count,
        COUNT(CASE WHEN is_active THEN 1 END) as active_count,
        ROUND(AVG(julianday('2024-04-01') - julianday(signup_date)), 1) as avg_days_since_signup
    FROM users
    GROUP BY tier
    ORDER BY user_count DESC
""", "User Statistics by Tier")

# EXERCISE 1.1: Write a query to find:
# - Total revenue per product category
# - Average order value per category
# - Number of orders per category
# Only include completed orders

"""
YOUR QUERY HERE:

SELECT
    ...
FROM order_items oi
JOIN ...
WHERE ...
GROUP BY ...
"""


# =============================================================================
# SECTION 2: JOINs - The Foundation of Data Engineering
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: JOINs")
print("=" * 70)

# --- 2.1 INNER JOIN ---

run_query(conn, """
    SELECT
        o.order_id,
        u.username,
        u.tier,
        o.order_date,
        o.total_amount,
        o.status
    FROM orders o
    INNER JOIN users u ON o.user_id = u.user_id
    WHERE o.status = 'completed'
    ORDER BY o.total_amount DESC
    LIMIT 10
""", "Top 10 Completed Orders with User Info")

# --- 2.2 LEFT JOIN (Finding Missing Data) ---

run_query(conn, """
    SELECT
        u.user_id,
        u.username,
        u.signup_date,
        COUNT(o.order_id) as order_count,
        COALESCE(SUM(o.total_amount), 0) as total_spent
    FROM users u
    LEFT JOIN orders o ON u.user_id = o.user_id
        AND o.status = 'completed'
    GROUP BY u.user_id, u.username, u.signup_date
    ORDER BY total_spent DESC
""", "All Users with Order Summary (including non-buyers)")

# --- 2.3 Multiple JOINs ---

run_query(conn, """
    SELECT
        u.username,
        u.tier,
        p.product_name,
        p.category,
        oi.quantity,
        oi.unit_price,
        (oi.quantity * oi.unit_price) as line_total
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN users u ON o.user_id = u.user_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE o.status = 'completed'
    ORDER BY line_total DESC
    LIMIT 10
""", "Detailed Order Items with User and Product Info")

# --- 2.4 Self JOIN (Comparing within same table) ---

run_query(conn, """
    SELECT
        u1.username as user1,
        u2.username as user2,
        u1.country
    FROM users u1
    JOIN users u2 ON u1.country = u2.country
        AND u1.user_id < u2.user_id  -- Avoid duplicates
    WHERE u1.is_active AND u2.is_active
    ORDER BY u1.country
""", "Users in Same Country (Self Join)")

# EXERCISE 2.1: Find products that have NEVER been ordered
# Hint: Use LEFT JOIN and check for NULL

"""
YOUR QUERY HERE:
"""


# =============================================================================
# SECTION 3: Window Functions - Interview Must-Know
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Window Functions")
print("=" * 70)

# --- 3.1 ROW_NUMBER, RANK, DENSE_RANK ---

run_query(conn, """
    SELECT
        user_id,
        order_id,
        order_date,
        total_amount,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date) as order_sequence,
        RANK() OVER (PARTITION BY user_id ORDER BY total_amount DESC) as amount_rank
    FROM orders
    WHERE status = 'completed'
    ORDER BY user_id, order_date
""", "Order Sequence and Ranking per User")

# --- 3.2 Running Totals and Moving Averages ---

run_query(conn, """
    SELECT
        user_id,
        order_date,
        total_amount,
        SUM(total_amount) OVER (
            PARTITION BY user_id
            ORDER BY order_date
            ROWS UNBOUNDED PRECEDING
        ) as cumulative_spend,
        AVG(total_amount) OVER (
            PARTITION BY user_id
            ORDER BY order_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) as moving_avg_3
    FROM orders
    WHERE status = 'completed'
    ORDER BY user_id, order_date
""", "Cumulative Spend and Moving Average")

# --- 3.3 LAG and LEAD (Previous/Next Values) ---

run_query(conn, """
    SELECT
        user_id,
        order_date,
        total_amount,
        LAG(total_amount, 1) OVER (PARTITION BY user_id ORDER BY order_date) as prev_order_amount,
        LEAD(order_date, 1) OVER (PARTITION BY user_id ORDER BY order_date) as next_order_date,
        julianday(order_date) - julianday(
            LAG(order_date, 1) OVER (PARTITION BY user_id ORDER BY order_date)
        ) as days_since_prev_order
    FROM orders
    WHERE status = 'completed'
    ORDER BY user_id, order_date
""", "Order Comparison with Previous Order")

# --- 3.4 FIRST_VALUE, LAST_VALUE ---

run_query(conn, """
    SELECT DISTINCT
        user_id,
        FIRST_VALUE(order_date) OVER (
            PARTITION BY user_id
            ORDER BY order_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) as first_order_date,
        LAST_VALUE(order_date) OVER (
            PARTITION BY user_id
            ORDER BY order_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) as last_order_date,
        FIRST_VALUE(total_amount) OVER (
            PARTITION BY user_id
            ORDER BY order_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) as first_order_amount
    FROM orders
    WHERE status = 'completed'
    ORDER BY user_id
""", "First and Last Order Info per User")

# EXERCISE 3.1: Using window functions, find:
# - Each user's largest order
# - What percentage of their total spend that order represents
# - The rank of that order among all orders

"""
YOUR QUERY HERE:
"""


# =============================================================================
# SECTION 4: CTEs and Subqueries - Code Organization
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: CTEs and Subqueries")
print("=" * 70)

# --- 4.1 Common Table Expressions (CTEs) ---
# PRODUCTION BEST PRACTICE: Use CTEs for readable, maintainable code

run_query(conn, """
    WITH user_metrics AS (
        -- Calculate user-level metrics
        SELECT
            u.user_id,
            u.username,
            u.tier,
            COUNT(o.order_id) as order_count,
            COALESCE(SUM(o.total_amount), 0) as total_revenue,
            MIN(o.order_date) as first_order,
            MAX(o.order_date) as last_order
        FROM users u
        LEFT JOIN orders o ON u.user_id = o.user_id
            AND o.status = 'completed'
        GROUP BY u.user_id, u.username, u.tier
    ),
    tier_averages AS (
        -- Calculate tier-level benchmarks
        SELECT
            tier,
            AVG(total_revenue) as tier_avg_revenue,
            AVG(order_count) as tier_avg_orders
        FROM user_metrics
        GROUP BY tier
    )
    SELECT
        um.username,
        um.tier,
        um.order_count,
        um.total_revenue,
        ROUND(ta.tier_avg_revenue, 2) as tier_avg_revenue,
        ROUND(um.total_revenue - ta.tier_avg_revenue, 2) as vs_tier_avg
    FROM user_metrics um
    JOIN tier_averages ta ON um.tier = ta.tier
    ORDER BY vs_tier_avg DESC
""", "User Performance vs Tier Average (using CTEs)")

# --- 4.2 Correlated Subqueries ---

run_query(conn, """
    SELECT
        u.user_id,
        u.username,
        (
            SELECT COUNT(*)
            FROM orders o
            WHERE o.user_id = u.user_id
                AND o.status = 'completed'
        ) as completed_orders,
        (
            SELECT COALESCE(SUM(total_amount), 0)
            FROM orders o
            WHERE o.user_id = u.user_id
                AND o.status = 'completed'
        ) as total_spent
    FROM users u
    WHERE u.is_active
    ORDER BY total_spent DESC
""", "User Stats using Correlated Subqueries")

# --- 4.3 EXISTS vs IN ---
# PRODUCTION TIP: EXISTS often performs better than IN for large datasets

run_query(conn, """
    -- Users who have made at least one purchase
    SELECT
        u.user_id,
        u.username,
        u.tier
    FROM users u
    WHERE EXISTS (
        SELECT 1
        FROM orders o
        WHERE o.user_id = u.user_id
            AND o.status = 'completed'
    )
""", "Users with Purchases (using EXISTS)")

# EXERCISE 4.1: Using CTEs, create a cohort analysis:
# - Group users by signup month
# - Calculate average revenue per cohort
# - Calculate retention (users who ordered in month after signup)

"""
YOUR QUERY HERE:
"""


# =============================================================================
# SECTION 5: Advanced Analytics Patterns
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Advanced Analytics")
print("=" * 70)

# --- 5.1 Funnel Analysis ---

run_query(conn, """
    WITH funnel AS (
        SELECT
            user_id,
            MAX(CASE WHEN event_type = 'page_view' THEN 1 ELSE 0 END) as viewed,
            MAX(CASE WHEN event_type = 'add_to_cart' THEN 1 ELSE 0 END) as added_to_cart,
            MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) as purchased
        FROM events
        GROUP BY user_id
    )
    SELECT
        COUNT(*) as total_users,
        SUM(viewed) as viewed_page,
        SUM(added_to_cart) as added_to_cart,
        SUM(purchased) as purchased,
        ROUND(100.0 * SUM(added_to_cart) / NULLIF(SUM(viewed), 0), 1) as view_to_cart_pct,
        ROUND(100.0 * SUM(purchased) / NULLIF(SUM(added_to_cart), 0), 1) as cart_to_purchase_pct,
        ROUND(100.0 * SUM(purchased) / NULLIF(SUM(viewed), 0), 1) as overall_conversion_pct
    FROM funnel
""", "Conversion Funnel Analysis")

# --- 5.2 Cohort Retention ---

run_query(conn, """
    WITH user_cohorts AS (
        SELECT
            user_id,
            strftime('%Y-%m', signup_date) as cohort_month
        FROM users
    ),
    user_orders AS (
        SELECT
            o.user_id,
            strftime('%Y-%m', o.order_date) as order_month
        FROM orders o
        WHERE o.status = 'completed'
    )
    SELECT
        uc.cohort_month,
        COUNT(DISTINCT uc.user_id) as cohort_size,
        COUNT(DISTINCT CASE
            WHEN uo.order_month = uc.cohort_month THEN uo.user_id
        END) as month_0_active,
        COUNT(DISTINCT CASE
            WHEN uo.order_month > uc.cohort_month THEN uo.user_id
        END) as retained_later
    FROM user_cohorts uc
    LEFT JOIN user_orders uo ON uc.user_id = uo.user_id
    GROUP BY uc.cohort_month
    ORDER BY uc.cohort_month
""", "Cohort Retention Analysis")

# --- 5.3 RFM Segmentation ---
# CLASSIC INTERVIEW QUESTION

run_query(conn, """
    WITH rfm_base AS (
        SELECT
            u.user_id,
            u.username,
            julianday('2024-04-01') - julianday(MAX(o.order_date)) as recency_days,
            COUNT(o.order_id) as frequency,
            COALESCE(SUM(o.total_amount), 0) as monetary
        FROM users u
        LEFT JOIN orders o ON u.user_id = o.user_id
            AND o.status = 'completed'
        GROUP BY u.user_id, u.username
    ),
    rfm_scores AS (
        SELECT
            user_id,
            username,
            recency_days,
            frequency,
            monetary,
            -- Score 1-5 (5 is best)
            CASE
                WHEN recency_days IS NULL THEN 1
                WHEN recency_days <= 30 THEN 5
                WHEN recency_days <= 60 THEN 4
                WHEN recency_days <= 90 THEN 3
                WHEN recency_days <= 180 THEN 2
                ELSE 1
            END as r_score,
            CASE
                WHEN frequency >= 4 THEN 5
                WHEN frequency >= 3 THEN 4
                WHEN frequency >= 2 THEN 3
                WHEN frequency >= 1 THEN 2
                ELSE 1
            END as f_score,
            CASE
                WHEN monetary >= 500 THEN 5
                WHEN monetary >= 300 THEN 4
                WHEN monetary >= 150 THEN 3
                WHEN monetary >= 50 THEN 2
                ELSE 1
            END as m_score
        FROM rfm_base
    )
    SELECT
        user_id,
        username,
        ROUND(recency_days, 0) as recency_days,
        frequency,
        ROUND(monetary, 2) as monetary,
        r_score || f_score || m_score as rfm_segment,
        CASE
            WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champion'
            WHEN r_score >= 4 AND f_score >= 3 THEN 'Loyal Customer'
            WHEN r_score >= 4 AND f_score <= 2 THEN 'New Customer'
            WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk'
            WHEN r_score <= 2 AND f_score <= 2 AND m_score >= 3 THEN 'Cant Lose'
            ELSE 'Need Attention'
        END as customer_segment
    FROM rfm_scores
    ORDER BY monetary DESC
""", "RFM Customer Segmentation")

# --- 5.4 Year-over-Year Comparison ---

run_query(conn, """
    WITH monthly_revenue AS (
        SELECT
            strftime('%Y', order_date) as year,
            strftime('%m', order_date) as month,
            SUM(total_amount) as revenue
        FROM orders
        WHERE status = 'completed'
        GROUP BY strftime('%Y', order_date), strftime('%m', order_date)
    )
    SELECT
        curr.month,
        curr.revenue as current_year,
        prev.revenue as previous_year,
        ROUND(
            100.0 * (curr.revenue - COALESCE(prev.revenue, 0)) /
            NULLIF(prev.revenue, 0),
            1
        ) as yoy_growth_pct
    FROM monthly_revenue curr
    LEFT JOIN monthly_revenue prev
        ON curr.month = prev.month
        AND CAST(curr.year AS INTEGER) = CAST(prev.year AS INTEGER) + 1
    WHERE curr.year = '2024'
    ORDER BY curr.month
""", "Year-over-Year Revenue Comparison")

# EXERCISE 5.1: Calculate Customer Lifetime Value (CLV)
# - Average order value per customer
# - Average orders per year (annualized)
# - Estimated CLV = AOV * Orders/Year * Estimated Years (use 3)

"""
YOUR QUERY HERE:
"""


# =============================================================================
# SECTION 6: Performance Optimization
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6: Performance Optimization")
print("=" * 70)

# --- 6.1 EXPLAIN Query Plan ---

print("--- Understanding Query Plans ---")
print("""
PRODUCTION TIPS:

1. Use EXPLAIN ANALYZE (PostgreSQL) or EXPLAIN (MySQL/SQLite) to understand query performance

2. Index Strategy:
   - Index columns used in WHERE, JOIN, ORDER BY
   - Composite indexes for multi-column filters
   - Don't over-index (slows writes)

3. Query Optimization:
   - Filter early (push WHERE conditions as early as possible)
   - Avoid SELECT * (specify needed columns)
   - Use EXISTS instead of IN for large subqueries
   - Avoid functions on indexed columns in WHERE

4. Common Anti-patterns:
   - WHERE YEAR(date_column) = 2024  -- Can't use index
   - WHERE column LIKE '%search%'    -- Can't use index (leading wildcard)
   - SELECT DISTINCT with large datasets
   - ORDER BY on non-indexed columns with large datasets
""")

# Example: Bad vs Good queries
print("\n--- Bad vs Good Query Patterns ---")
print("""
-- BAD: Function on indexed column prevents index usage
SELECT * FROM orders WHERE YEAR(order_date) = 2024;

-- GOOD: Range condition can use index
SELECT * FROM orders WHERE order_date >= '2024-01-01' AND order_date < '2025-01-01';

-- BAD: SELECT * returns unnecessary data
SELECT * FROM orders o JOIN users u ON o.user_id = u.user_id;

-- GOOD: Select only needed columns
SELECT o.order_id, o.total_amount, u.username
FROM orders o JOIN users u ON o.user_id = u.user_id;

-- BAD: Subquery executed for each row
SELECT *, (SELECT COUNT(*) FROM orders WHERE user_id = u.user_id) as order_count
FROM users u;

-- GOOD: Single aggregation with join
SELECT u.*, COALESCE(o.order_count, 0) as order_count
FROM users u
LEFT JOIN (SELECT user_id, COUNT(*) as order_count FROM orders GROUP BY user_id) o
ON u.user_id = o.user_id;
""")


# =============================================================================
# SECTION 7: Interview Questions & Exercises
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 7: Interview Exercises")
print("=" * 70)

print("""
=== EXERCISE 7.1: Second Highest Salary ===
Classic interview question!
Write a query to find the second highest total_amount from orders.
Handle the case where there might be ties.

Expected approach: Use DENSE_RANK or subquery
""")

# Solution:
run_query(conn, """
    WITH ranked_orders AS (
        SELECT
            order_id,
            total_amount,
            DENSE_RANK() OVER (ORDER BY total_amount DESC) as rank
        FROM orders
        WHERE status = 'completed'
    )
    SELECT total_amount as second_highest_amount
    FROM ranked_orders
    WHERE rank = 2
    LIMIT 1
""", "Solution 7.1: Second Highest Order Amount")


print("""
=== EXERCISE 7.2: Consecutive Days ===
Find users who have placed orders on at least 2 consecutive days.
Return user_id, username, and the consecutive dates.
""")

# Solution:
run_query(conn, """
    WITH ordered_dates AS (
        SELECT DISTINCT
            o.user_id,
            u.username,
            o.order_date,
            LAG(o.order_date) OVER (PARTITION BY o.user_id ORDER BY o.order_date) as prev_date
        FROM orders o
        JOIN users u ON o.user_id = u.user_id
        WHERE o.status = 'completed'
    )
    SELECT
        user_id,
        username,
        prev_date as day_1,
        order_date as day_2
    FROM ordered_dates
    WHERE julianday(order_date) - julianday(prev_date) = 1
""", "Solution 7.2: Users with Consecutive Order Days")


print("""
=== EXERCISE 7.3: Running Total with Reset ===
Calculate running total of orders per user, but reset when total exceeds $500.
This tests advanced window function understanding.
""")

# This requires a different approach - recursive CTE or procedural logic
print("Note: This problem typically requires recursive CTEs or procedural code.")
print("Here's a simpler version - running total without reset:")

run_query(conn, """
    SELECT
        user_id,
        order_date,
        total_amount,
        SUM(total_amount) OVER (
            PARTITION BY user_id
            ORDER BY order_date
            ROWS UNBOUNDED PRECEDING
        ) as running_total,
        CASE
            WHEN SUM(total_amount) OVER (
                PARTITION BY user_id
                ORDER BY order_date
                ROWS UNBOUNDED PRECEDING
            ) > 500 THEN 'THRESHOLD_EXCEEDED'
            ELSE 'UNDER_THRESHOLD'
        END as status
    FROM orders
    WHERE status = 'completed'
    ORDER BY user_id, order_date
""", "Running Total with Threshold Flag")


print("""
=== EXERCISE 7.4: Gap Analysis ===
Find the gaps in order_id sequence (missing order IDs).
Common for data quality checks.
""")

run_query(conn, """
    WITH order_ids AS (
        SELECT
            order_id,
            LEAD(order_id) OVER (ORDER BY order_id) as next_id
        FROM orders
    )
    SELECT
        order_id as gap_start,
        next_id as gap_end,
        next_id - order_id - 1 as missing_count
    FROM order_ids
    WHERE next_id - order_id > 1
""", "Solution 7.4: Finding Gaps in Order IDs")


print("""
=== EXERCISE 7.5: Percentile Calculation ===
Calculate the 25th, 50th (median), and 75th percentile of order amounts.
""")

run_query(conn, """
    WITH ordered_amounts AS (
        SELECT
            total_amount,
            NTILE(4) OVER (ORDER BY total_amount) as quartile
        FROM orders
        WHERE status = 'completed'
    )
    SELECT
        quartile,
        MIN(total_amount) as min_in_quartile,
        MAX(total_amount) as max_in_quartile,
        ROUND(AVG(total_amount), 2) as avg_in_quartile
    FROM ordered_amounts
    GROUP BY quartile
    ORDER BY quartile
""", "Solution 7.5: Quartile Analysis")


# =============================================================================
# SECTION 8: Production SQL Patterns
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 8: Production Patterns")
print("=" * 70)

print("""
=== 8.1 Idempotent Data Loads ===
Use MERGE/UPSERT patterns for production ETL.

-- PostgreSQL example
INSERT INTO target_table (id, value, updated_at)
SELECT id, value, NOW()
FROM source_table
ON CONFLICT (id)
DO UPDATE SET
    value = EXCLUDED.value,
    updated_at = EXCLUDED.updated_at;

=== 8.2 Incremental Processing ===
Process only new/changed data.

-- Using watermark pattern
SELECT *
FROM events
WHERE event_timestamp > (
    SELECT COALESCE(MAX(processed_timestamp), '1970-01-01')
    FROM etl_watermarks
    WHERE table_name = 'events'
);

=== 8.3 Slowly Changing Dimensions (SCD Type 2) ===
Track historical changes to dimension data.

-- Check for changes and insert new version
INSERT INTO users_history (user_id, tier, valid_from, valid_to, is_current)
SELECT
    s.user_id,
    s.tier,
    CURRENT_DATE as valid_from,
    NULL as valid_to,
    TRUE as is_current
FROM users_staging s
JOIN users_history h ON s.user_id = h.user_id AND h.is_current = TRUE
WHERE s.tier != h.tier;

-- Close old records
UPDATE users_history
SET valid_to = CURRENT_DATE - 1, is_current = FALSE
WHERE user_id IN (...) AND is_current = TRUE;

=== 8.4 Data Quality Checks ===
Build automated quality gates.

-- Null check
SELECT 'user_id nulls' as check_name,
       COUNT(*) as violation_count
FROM orders WHERE user_id IS NULL

UNION ALL

-- Referential integrity
SELECT 'orphan orders' as check_name,
       COUNT(*) as violation_count
FROM orders o
LEFT JOIN users u ON o.user_id = u.user_id
WHERE u.user_id IS NULL

UNION ALL

-- Business rule: no negative amounts
SELECT 'negative amounts' as check_name,
       COUNT(*) as violation_count
FROM orders WHERE total_amount < 0;
""")


# =============================================================================
# FINAL: Practice Problems
# =============================================================================

print("\n" + "=" * 70)
print("PRACTICE PROBLEMS FOR INTERVIEW PREP")
print("=" * 70)

print("""
Try these problems on your own, then check with the database:

1. REVENUE TRENDS
   Calculate month-over-month revenue growth rate for completed orders.

2. TOP N PER GROUP
   Find the top 2 highest-spending customers per country.

3. MEDIAN CALCULATION
   Calculate the median order amount (not using built-in functions).

4. SESSIONIZATION
   Group events into sessions (new session if > 30 min gap).

5. ATTRIBUTION
   For each purchase event, find the first page the user visited in that session.

6. ACTIVE USERS
   Calculate Daily Active Users (DAU) and Monthly Active Users (MAU) ratio.

7. CHURN PREDICTION FEATURES
   Create features for churn prediction:
   - Days since last order
   - Order frequency (orders per month)
   - Average order value
   - Trend (is AOV increasing or decreasing?)

8. DUPLICATE DETECTION
   Find duplicate orders (same user, same day, same amount).

Good luck with your interviews!
""")

# Close connection
conn.close()

print("\n" + "=" * 70)
print("Tutorial Complete!")
print("=" * 70)
