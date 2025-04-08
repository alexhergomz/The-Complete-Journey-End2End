#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Initial Data Exploration for Dunnhumby The Complete Journey dataset using SQLite
Part of Phase 1.2: Initial Data Extraction and Exploration with SQL
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Set file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'dunnhumby.db')

# Create a connection to the database
def connect_to_db():
    """Create a connection to the SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        print("Connected to the database successfully!")
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

# Execute SQL query and return results as a pandas DataFrame
def execute_query(conn, query):
    """Execute a SQL query and return results as a pandas DataFrame"""
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error executing query: {e}")
        return None

# Main exploration function
def explore_data():
    """Perform initial data exploration"""
    conn = connect_to_db()
    if not conn:
        return
    
    # Create output directory for plots
    os.makedirs('exploration_results', exist_ok=True)
    
    # 1. Basic transaction summary
    print("\n1. Basic Transaction Summary")
    query = """
    SELECT 
        COUNT(DISTINCT household_key) AS unique_customers,
        COUNT(DISTINCT basket_id) AS total_baskets,
        SUM(sales_value) AS total_sales,
        AVG(sales_value) AS avg_basket_value
    FROM transaction_data;
    """
    results = execute_query(conn, query)
    display(results)
    
    # 2. Product category performance
    print("\n2. Product Category Performance")
    query = """
    SELECT 
        p.department AS department, 
        COUNT(*) AS transaction_count,
        SUM(t.sales_value) AS total_sales,
        ROUND(AVG(t.sales_value), 2) AS avg_sales_value,
        COUNT(DISTINCT t.household_key) AS unique_customers
    FROM transaction_data t
    JOIN product p ON t.product_id = p.product_id
    GROUP BY p.department
    ORDER BY total_sales DESC;
    """
    results = execute_query(conn, query)
    display(results)
    
    # Visualize top departments by sales
    plt.figure(figsize=(12, 6))
    
    # Add debug print to see actual column names
    print("DataFrame columns:", results.columns.tolist())
    
    # Use the top 10 departments
    top_departments = results.head(10)
    sns.barplot(x='total_sales', y='department', data=top_departments)
    plt.title('Top 10 Departments by Sales')
    plt.tight_layout()
    plt.savefig('exploration_results/top_departments_sales.png')
    
    # 3. Customer purchasing patterns
    print("\n3. Customer Purchasing Patterns")
    query = """
    SELECT 
        household_key,
        COUNT(DISTINCT basket_id) AS basket_count,
        COUNT(DISTINCT product_id) AS unique_products,
        SUM(sales_value) AS total_spend,
        ROUND(AVG(sales_value), 2) AS avg_item_spend,
        MIN(day) AS first_day,
        MAX(day) AS last_day,
        MAX(day) - MIN(day) AS day_span
    FROM transaction_data
    GROUP BY household_key
    ORDER BY total_spend DESC
    LIMIT 20;
    """
    results = execute_query(conn, query)
    display(results)
    
    # 4. Temporal patterns
    print("\n4. Temporal Patterns")
    query = """
    SELECT 
        week_no,
        COUNT(DISTINCT basket_id) AS basket_count,
        COUNT(DISTINCT household_key) AS customer_count,
        SUM(sales_value) AS total_sales,
        ROUND(AVG(sales_value), 2) AS avg_basket_value
    FROM transaction_data
    GROUP BY week_no
    ORDER BY week_no;
    """
    results = execute_query(conn, query)
    display(results)
    
    # Visualize weekly sales trend
    plt.figure(figsize=(14, 6))
    plt.plot(results['week_no'], results['total_sales'])
    plt.title('Weekly Sales Trend')
    plt.xlabel('Week Number')
    plt.ylabel('Total Sales')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exploration_results/weekly_sales_trend.png')
    
    # 5. Campaign effectiveness
    print("\n5. Campaign Effectiveness")
    query = """
    WITH campaign_stats AS (
        SELECT 
            cd.campaign,
            cd.description,
            COUNT(DISTINCT ct.household_key) AS total_households,
            cd.start_day,
            cd.end_day,
            cd.end_day - cd.start_day AS campaign_duration
        FROM campaign_desc cd
        JOIN campaign_table ct ON cd.campaign = ct.campaign
        GROUP BY cd.campaign, cd.description, cd.start_day, cd.end_day
    ),
    campaign_sales AS (
        SELECT 
            cd.campaign,
            SUM(t.sales_value) AS campaign_sales,
            COUNT(DISTINCT t.basket_id) AS basket_count
        FROM transaction_data t
        JOIN campaign_table ct ON t.household_key = ct.household_key
        JOIN campaign_desc cd ON ct.campaign = cd.campaign
        WHERE t.day BETWEEN cd.start_day AND cd.end_day
        GROUP BY cd.campaign
    )
    SELECT 
        cs.campaign,
        cst.description,
        cst.total_households,
        cst.campaign_duration,
        cs.campaign_sales,
        cs.basket_count,
        ROUND(cs.campaign_sales / cst.total_households, 2) AS sales_per_household
    FROM campaign_stats cst
    JOIN campaign_sales cs ON cst.campaign = cs.campaign
    ORDER BY sales_per_household DESC;
    """
    results = execute_query(conn, query)
    display(results)
    
    # 6. Demographic analysis
    print("\n6. Demographic Analysis")
    query = """
    SELECT 
        hd.age_desc AS age_desc,
        COUNT(DISTINCT hd.household_key) AS household_count,
        COUNT(DISTINCT t.basket_id) AS basket_count,
        ROUND(SUM(t.sales_value), 2) AS total_sales,
        ROUND(SUM(t.sales_value) / COUNT(DISTINCT hd.household_key), 2) AS sales_per_household,
        ROUND(COUNT(DISTINCT t.basket_id) * 1.0 / COUNT(DISTINCT hd.household_key), 2) AS baskets_per_household
    FROM household_demographics hd
    JOIN transaction_data t ON hd.household_key = t.household_key
    GROUP BY hd.age_desc
    ORDER BY sales_per_household DESC;
    """
    results = execute_query(conn, query)
    display(results)
    
    # Visualize sales by age group
    plt.figure(figsize=(10, 6))
    
    # Add debug print to see actual column names
    print("Demographic DataFrame columns:", results.columns.tolist())
    
    sns.barplot(x='age_desc', y='sales_per_household', data=results)
    plt.title('Sales per Household by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Sales per Household ($)')
    plt.tight_layout()
    plt.savefig('exploration_results/sales_by_age.png')
    
    # 7. Income level analysis
    print("\n7. Income Level Analysis")
    query = """
    SELECT 
        hd.income_desc,
        COUNT(DISTINCT hd.household_key) AS household_count,
        ROUND(SUM(t.sales_value), 2) AS total_sales,
        ROUND(SUM(t.sales_value) / COUNT(DISTINCT hd.household_key), 2) AS sales_per_household,
        ROUND(COUNT(DISTINCT t.basket_id) * 1.0 / COUNT(DISTINCT hd.household_key), 2) AS baskets_per_household
    FROM household_demographics hd
    JOIN transaction_data t ON hd.household_key = t.household_key
    GROUP BY hd.income_desc
    ORDER BY 
        CASE 
            WHEN hd.income_desc = 'Under 15K' THEN 1
            WHEN hd.income_desc = '15-24K' THEN 2
            WHEN hd.income_desc = '25-34K' THEN 3
            WHEN hd.income_desc = '35-49K' THEN 4
            WHEN hd.income_desc = '50-74K' THEN 5
            WHEN hd.income_desc = '75-99K' THEN 6
            WHEN hd.income_desc = '100-124K' THEN 7
            WHEN hd.income_desc = '125-149K' THEN 8
            WHEN hd.income_desc = '150-174K' THEN 9
            WHEN hd.income_desc = '175-199K' THEN 10
            WHEN hd.income_desc = '200-249K' THEN 11
            WHEN hd.income_desc = '250K+' THEN 12
            ELSE 13
        END;
    """
    results = execute_query(conn, query)
    display(results)
    
    # 8. Coupon effectiveness
    print("\n8. Coupon Effectiveness")
    query = """
    WITH coupon_stats AS (
        SELECT 
            c.campaign,
            COUNT(DISTINCT cr.household_key) AS redeeming_households,
            COUNT(*) AS redemption_count
        FROM coupon_redemption cr
        JOIN coupon c ON cr.coupon_upc = c.coupon_upc AND cr.campaign = c.campaign
        GROUP BY c.campaign
    ),
    campaign_households AS (
        SELECT 
            campaign,
            COUNT(DISTINCT household_key) AS targeted_households
        FROM campaign_table
        GROUP BY campaign
    )
    SELECT 
        cd.campaign,
        cd.description,
        ch.targeted_households,
        COALESCE(cs.redeeming_households, 0) AS redeeming_households,
        COALESCE(cs.redemption_count, 0) AS redemption_count,
        CASE 
            WHEN ch.targeted_households > 0 THEN 
                ROUND(COALESCE(cs.redeeming_households, 0) * 100.0 / ch.targeted_households, 2)
            ELSE 0
        END AS redemption_rate_pct
    FROM campaign_desc cd
    LEFT JOIN campaign_households ch ON cd.campaign = ch.campaign
    LEFT JOIN coupon_stats cs ON cd.campaign = cs.campaign
    ORDER BY redemption_rate_pct DESC;
    """
    results = execute_query(conn, query)
    display(results)
    
    # Close the database connection
    conn.close()
    print("\nData exploration completed. Check the 'exploration_results' folder for visualizations.")

if __name__ == "__main__":
    explore_data() 