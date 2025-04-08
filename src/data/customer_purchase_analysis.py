#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Customer Purchase Behavior Analysis
Identifying significant variables that influence purchase behavior through:
- Correlation analysis
- Statistical significance testing
- Feature selection
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from IPython.display import display

# Set file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'dunnhumby.db')
RESULTS_DIR = os.path.join(BASE_DIR, 'analysis_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

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

def create_customer_features():
    """Create a comprehensive feature set for each customer"""
    conn = connect_to_db()
    if not conn:
        return None
    
    print("Building customer feature dataset...")
    
    # Query to create comprehensive customer features
    query = """
    WITH customer_transactions AS (
        -- Basic transaction metrics per customer
        SELECT
            t.household_key,
            COUNT(DISTINCT t.basket_id) AS total_baskets,
            COUNT(DISTINCT t.product_id) AS unique_products,
            SUM(t.quantity) AS total_items,
            SUM(t.sales_value) AS total_spend,
            AVG(t.sales_value) AS avg_transaction_value,
            MAX(t.day) - MIN(t.day) AS day_span,
            COUNT(DISTINCT t.week_no) AS active_weeks,
            SUM(t.retail_disc) AS total_retail_discount,
            SUM(t.coupon_disc) AS total_coupon_discount,
            SUM(t.coupon_match_disc) AS total_coupon_match_discount,
            COUNT(DISTINCT t.store_id) AS unique_stores_visited
        FROM transaction_data t
        GROUP BY t.household_key
    ),
    
    product_categories AS (
        -- Product category preferences
        SELECT
            t.household_key,
            p.department,
            COUNT(*) AS dept_transactions,
            SUM(t.sales_value) AS dept_spend
        FROM transaction_data t
        JOIN product p ON t.product_id = p.product_id
        GROUP BY t.household_key, p.department
    ),
    
    category_preferences AS (
        -- Calculate top department for each customer
        SELECT
            household_key,
            department AS top_department,
            dept_spend
        FROM (
            SELECT
                household_key,
                department,
                dept_spend,
                ROW_NUMBER() OVER (PARTITION BY household_key ORDER BY dept_spend DESC) as rn
            FROM product_categories
        )
        WHERE rn = 1
    ),
    
    campaign_participation AS (
        -- Campaign engagement metrics
        SELECT
            ct.household_key,
            COUNT(DISTINCT ct.campaign) AS campaigns_participated,
            COUNT(DISTINCT cr.coupon_upc) AS coupons_redeemed
        FROM campaign_table ct
        LEFT JOIN coupon_redemption cr ON ct.household_key = cr.household_key
        GROUP BY ct.household_key
    ),
    
    purchase_frequency AS (
        -- Calculate purchase frequency metrics
        SELECT
            household_key,
            COUNT(DISTINCT basket_id) AS total_trips,
            COUNT(DISTINCT day) AS unique_days,
            CASE 
                WHEN MAX(day) - MIN(day) > 0 THEN 
                    CAST(COUNT(DISTINCT basket_id) AS FLOAT) / (MAX(day) - MIN(day))
                ELSE NULL
            END AS basket_frequency
        FROM transaction_data
        GROUP BY household_key
    )
    
    SELECT
        ct.*,
        COALESCE(cp.top_department, 'UNKNOWN') AS top_department,
        COALESCE(cp.dept_spend, 0) AS top_dept_spend,
        COALESCE(cmp.campaigns_participated, 0) AS campaigns_participated,
        COALESCE(cmp.coupons_redeemed, 0) AS coupons_redeemed,
        pf.basket_frequency,
        hd.age_desc,
        hd.marital_status_code,
        hd.income_desc,
        hd.homeowner_desc,
        hd.hh_comp_desc,
        hd.household_size_desc,
        hd.kid_category_desc
    FROM customer_transactions ct
    LEFT JOIN category_preferences cp ON ct.household_key = cp.household_key
    LEFT JOIN campaign_participation cmp ON ct.household_key = cmp.household_key
    LEFT JOIN purchase_frequency pf ON ct.household_key = pf.household_key
    LEFT JOIN household_demographics hd ON ct.household_key = hd.household_key
    """
    
    customer_features = execute_query(conn, query)
    
    if customer_features is not None:
        print(f"Successfully created feature dataset with {len(customer_features)} customers and {customer_features.shape[1]} features.")
        
        # Save the raw features data
        customer_features.to_csv(os.path.join(RESULTS_DIR, 'customer_features.csv'), index=False)
    
    conn.close()
    return customer_features

def prepare_features(customer_features):
    """Prepare features for analysis by handling missing values and encoding categorical variables"""
    if customer_features is None:
        return None
    
    print("Preparing features for analysis...")
    
    # Make a copy to avoid modifying the original
    df = customer_features.copy()
    
    # Handle missing values
    df['basket_frequency'].fillna(0, inplace=True)
    
    # Encode categorical variables
    categorical_cols = ['age_desc', 'marital_status_code', 'income_desc', 'homeowner_desc', 
                        'hh_comp_desc', 'household_size_desc', 'kid_category_desc', 'top_department']
    
    # Drop rows with missing demographic data if needed
    # df = df.dropna(subset=['age_desc', 'income_desc'])  # Uncomment if you want to drop missing demographics
    
    # One-hot encode categorical columns
    for col in categorical_cols:
        if col in df.columns:
            # Fill NAs with 'UNKNOWN' before encoding
            df[col].fillna('UNKNOWN', inplace=True)
            # Get dummies and drop the original column
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
    
    # Create derived metrics
    df['discount_ratio'] = np.where(df['total_spend'] > 0, 
                                  (df['total_retail_discount'] + df['total_coupon_discount'] + df['total_coupon_match_discount']) / df['total_spend'], 
                                  0)
    
    df['avg_basket_value'] = np.where(df['total_baskets'] > 0, df['total_spend'] / df['total_baskets'], 0)
    df['avg_items_per_basket'] = np.where(df['total_baskets'] > 0, df['total_items'] / df['total_baskets'], 0)
    df['product_diversity'] = np.where(df['total_items'] > 0, df['unique_products'] / df['total_items'], 0)
    df['coupon_redemption_rate'] = np.where(df['campaigns_participated'] > 0, df['coupons_redeemed'] / df['campaigns_participated'], 0)
    
    print(f"Feature preparation complete: {df.shape[1]} processed features.")
    return df

def correlation_analysis(df):
    """Perform correlation analysis on customer features"""
    print("Performing correlation analysis...")
    
    # Select numeric columns for correlation analysis
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Drop identifier columns and highly correlated columns to avoid redundancy
    if 'household_key' in numeric_cols:
        numeric_cols.remove('household_key')
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Save correlation matrix
    corr_matrix.to_csv(os.path.join(RESULTS_DIR, 'correlation_matrix.csv'))
    
    # Visualize correlation matrix (top 20 features vs total_spend)
    target_var = 'total_spend'
    correlations = corr_matrix[target_var].sort_values(ascending=False)
    
    # Select top and bottom 10 correlations (excluding the target itself)
    top_features = correlations[1:11].index.tolist()  # Exclude target itself (position 0)
    bottom_features = correlations[-10:].index.tolist()
    selected_features = top_features + bottom_features
    
    # Create correlation plot
    plt.figure(figsize=(10, 12))
    corr_with_target = corr_matrix.loc[selected_features, [target_var]]
    sns.heatmap(corr_with_target, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f'Correlation with {target_var}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'target_correlation_heatmap.png'))
    
    # Create full correlation heatmap for top features
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix.loc[top_features, top_features], annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix for Top Features')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_correlation_heatmap.png'))
    
    print("Correlation analysis completed.")
    return corr_matrix, top_features

def significance_testing(df, top_features):
    """Perform statistical significance testing for key features"""
    print("Performing statistical significance testing...")
    
    target_var = 'total_spend'
    
    # Build a DataFrame to store results
    results = pd.DataFrame(columns=['Feature', 'Correlation', 'P-Value', 'Significant'])
    
    # Perform significance testing for each feature
    for feature in top_features:
        # Calculate Pearson correlation and p-value
        correlation, p_value = stats.pearsonr(df[feature], df[target_var])
        
        # Add to results
        results = pd.concat([results, pd.DataFrame({
            'Feature': [feature],
            'Correlation': [correlation],
            'P-Value': [p_value],
            'Significant': [p_value < 0.05]
        })], ignore_index=True)
    
    # Sort by absolute correlation
    results = results.sort_values(by='Correlation', key=abs, ascending=False)
    
    # Save results
    results.to_csv(os.path.join(RESULTS_DIR, 'significance_test_results.csv'), index=False)
    
    # Display significant features
    print("\nStatistically significant features:")
    significant_features = results[results['Significant']].sort_values(by='Correlation', key=abs, ascending=False)
    
    # Create a bar plot of significant correlations
    plt.figure(figsize=(12, 8))
    plt.barh(significant_features['Feature'], significant_features['Correlation'])
    plt.xlabel('Correlation with Total Spend')
    plt.ylabel('Feature')
    plt.title('Significant Features Correlation with Total Spend')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'significant_features.png'))
    
    print("Significance testing completed.")
    return significant_features

def feature_importance_analysis(df, top_features):
    """Analyze feature importance using regression-based methods"""
    print("Analyzing feature importance...")
    
    target_var = 'total_spend'
    
    # Create a subset of data with selected features
    X = df[top_features]
    y = df[target_var]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=top_features)
    
    # 1. Linear regression with feature coefficients
    X_with_const = sm.add_constant(X_scaled_df)
    model = sm.OLS(y, X_with_const).fit()
    
    # Extract coefficients and p-values
    coef_df = pd.DataFrame({
        'Feature': top_features,
        'Coefficient': model.params.values[1:],  # Skip the constant
        'P-Value': model.pvalues.values[1:],  # Skip the constant
        'Significant': model.pvalues.values[1:] < 0.05  # Skip the constant
    })
    
    # Sort by absolute coefficient value
    coef_df = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)
    
    # Save regression results
    with open(os.path.join(RESULTS_DIR, 'regression_summary.txt'), 'w') as f:
        f.write(model.summary().as_text())
    
    coef_df.to_csv(os.path.join(RESULTS_DIR, 'feature_coefficients.csv'), index=False)
    
    # 2. Calculate VIF (Variance Inflation Factor) to detect multicollinearity
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X_with_const.columns[1:]  # Skip the constant
    vif_data['VIF'] = [variance_inflation_factor(X_with_const.values, i) for i in range(1, X_with_const.shape[1])]
    vif_data = vif_data.sort_values('VIF', ascending=False)
    vif_data.to_csv(os.path.join(RESULTS_DIR, 'vif_results.csv'), index=False)
    
    # Create coefficient plot for significant features
    significant_coef = coef_df[coef_df['Significant']]
    plt.figure(figsize=(12, 8))
    plt.barh(significant_coef['Feature'], significant_coef['Coefficient'])
    plt.xlabel('Standardized Coefficient')
    plt.ylabel('Feature')
    plt.title('Significant Feature Coefficients (Standardized)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'significant_coefficients.png'))
    
    print("Feature importance analysis completed.")
    return coef_df, vif_data

def main():
    """Main analysis function"""
    print("Starting customer purchase behavior analysis...")
    
    # Create customer features dataset
    customer_features = create_customer_features()
    
    if customer_features is None:
        print("Error: Could not create customer features dataset.")
        return
    
    # Prepare features for analysis
    processed_features = prepare_features(customer_features)
    
    if processed_features is None:
        print("Error: Could not process features.")
        return
    
    # Perform correlation analysis
    corr_matrix, top_features = correlation_analysis(processed_features)
    
    # Perform significance testing
    significant_features = significance_testing(processed_features, top_features)
    
    # Analyze feature importance
    feature_coefficients, vif_results = feature_importance_analysis(processed_features, top_features)
    
    # Generate insights summary
    print("\n===== ANALYSIS SUMMARY =====")
    print(f"Total customers analyzed: {len(processed_features)}")
    print(f"Total features considered: {processed_features.shape[1]}")
    print(f"Top correlated features with spending: {', '.join(top_features[:5])}")
    
    # Print top 5 significant features
    significant_list = significant_features.head(5)['Feature'].tolist()
    print(f"Top 5 statistically significant features: {', '.join(significant_list)}")
    
    # Print potential use cases
    print("\nPotential applications of these insights:")
    print("1. Customer segmentation based on identified significant variables")
    print("2. Targeted marketing campaigns for high-value customer segments")
    print("3. Personalized promotions based on purchase patterns")
    print("4. Customer lifetime value prediction models")
    print("5. Churn prediction and prevention strategies")
    
    print("\nAnalysis completed successfully. Results saved to the 'analysis_results' directory.")

if __name__ == "__main__":
    main() 