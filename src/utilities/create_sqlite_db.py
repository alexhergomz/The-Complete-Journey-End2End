#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SQLite Database Setup for Dunnhumby The Complete Journey dataset
Part of Phase 1.1: Database Setup with SQL
"""

import os
import sqlite3
import csv
import pandas as pd
import time

# Set file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(BASE_DIR, 'archive')
DB_PATH = os.path.join(BASE_DIR, 'dunnhumby.db')

# CSV file paths
CSV_FILES = {
    'household_demographics': os.path.join(ARCHIVE_DIR, 'hh_demographic.csv'),
    'product': os.path.join(ARCHIVE_DIR, 'product.csv'),
    'campaign_desc': os.path.join(ARCHIVE_DIR, 'campaign_desc.csv'),
    'campaign_table': os.path.join(ARCHIVE_DIR, 'campaign_table.csv'),
    'coupon': os.path.join(ARCHIVE_DIR, 'coupon.csv'),
    'coupon_redemption': os.path.join(ARCHIVE_DIR, 'coupon_redempt.csv'),
    'transaction_data': os.path.join(ARCHIVE_DIR, 'transaction_data.csv'),
    'causal_data': os.path.join(ARCHIVE_DIR, 'causal_data.csv')
}

# Create SQLite database schema
def create_database_schema():
    """Create the SQLite database schema"""
    
    print("Creating database schema...")
    
    # Connect to SQLite database (will create it if it doesn't exist)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables
    
    # Household demographics table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS household_demographics (
        household_key INTEGER PRIMARY KEY,
        age_desc TEXT,
        marital_status_code TEXT,
        income_desc TEXT,
        homeowner_desc TEXT,
        hh_comp_desc TEXT,
        household_size_desc TEXT,
        kid_category_desc TEXT
    )
    ''')
    
    # Product table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS product (
        product_id INTEGER PRIMARY KEY,
        manufacturer INTEGER,
        department TEXT,
        brand TEXT,
        commodity_desc TEXT,
        sub_commodity_desc TEXT,
        curr_size_of_product TEXT
    )
    ''')
    
    # Campaign description table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS campaign_desc (
        campaign INTEGER PRIMARY KEY,
        description TEXT,
        start_day INTEGER,
        end_day INTEGER
    )
    ''')
    
    # Campaign table (household-campaign mapping)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS campaign_table (
        household_key INTEGER,
        campaign INTEGER,
        description TEXT,
        PRIMARY KEY (household_key, campaign),
        FOREIGN KEY (household_key) REFERENCES household_demographics(household_key),
        FOREIGN KEY (campaign) REFERENCES campaign_desc(campaign)
    )
    ''')
    
    # Coupon table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS coupon (
        coupon_upc TEXT,
        product_id INTEGER,
        campaign INTEGER,
        PRIMARY KEY (coupon_upc, product_id, campaign),
        FOREIGN KEY (product_id) REFERENCES product(product_id),
        FOREIGN KEY (campaign) REFERENCES campaign_desc(campaign)
    )
    ''')
    
    # Coupon redemption table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS coupon_redemption (
        household_key INTEGER,
        day INTEGER,
        coupon_upc TEXT,
        campaign INTEGER,
        PRIMARY KEY (household_key, day, coupon_upc),
        FOREIGN KEY (household_key) REFERENCES household_demographics(household_key)
    )
    ''')
    
    # Transaction data table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transaction_data (
        household_key INTEGER,
        basket_id INTEGER,
        day INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        sales_value REAL,
        store_id INTEGER,
        retail_disc REAL,
        trans_time INTEGER,
        week_no INTEGER,
        coupon_disc REAL,
        coupon_match_disc REAL,
        PRIMARY KEY (basket_id, household_key, product_id),
        FOREIGN KEY (household_key) REFERENCES household_demographics(household_key),
        FOREIGN KEY (product_id) REFERENCES product(product_id)
    )
    ''')
    
    # Causal data table (for promotions and displays)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS causal_data (
        product_id INTEGER,
        store_id INTEGER,
        week_no INTEGER,
        display TEXT,
        mailer TEXT,
        PRIMARY KEY (product_id, store_id, week_no),
        FOREIGN KEY (product_id) REFERENCES product(product_id)
    )
    ''')
    
    # Create indexes for faster querying
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_transaction_household ON transaction_data(household_key)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_transaction_product ON transaction_data(product_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_transaction_week ON transaction_data(week_no)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_transaction_day ON transaction_data(day)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_coupon_redemption_household ON coupon_redemption(household_key)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_coupon_redemption_campaign ON coupon_redemption(campaign)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_campaign_table_household ON campaign_table(household_key)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_campaign_table_campaign ON campaign_table(campaign)')
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print("Database schema created successfully.")

def import_data():
    """Import data from CSV files into SQLite database"""
    
    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    
    # Set the smaller tables to import first
    small_tables = [
        ('household_demographics', CSV_FILES['household_demographics']),
        ('product', CSV_FILES['product']),
        ('campaign_desc', CSV_FILES['campaign_desc']),
        ('campaign_table', CSV_FILES['campaign_table']),
        ('coupon', CSV_FILES['coupon']),
        ('coupon_redemption', CSV_FILES['coupon_redemption'])
    ]
    
    # Import smaller tables
    for table_name, csv_file in small_tables:
        print(f"Importing {table_name} data...")
        
        # Read CSV file with pandas and insert into SQLite
        df = pd.read_csv(csv_file)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        print(f"Imported {len(df)} rows into {table_name}.")
    
    # Import large tables with chunking for better memory management
    large_tables = [
        ('transaction_data', CSV_FILES['transaction_data']),
        ('causal_data', CSV_FILES['causal_data'])
    ]
    
    for table_name, csv_file in large_tables:
        print(f"Importing {table_name} data (this may take some time)...")
        start_time = time.time()
        
        # Use chunksize for large files
        chunksize = 100000
        total_rows = 0
        
        # Drop the table first for better performance
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Recreate table for transaction_data
        if table_name == 'transaction_data':
            conn.execute('''
            CREATE TABLE IF NOT EXISTS transaction_data (
                household_key INTEGER,
                basket_id INTEGER,
                day INTEGER,
                product_id INTEGER,
                quantity INTEGER,
                sales_value REAL,
                store_id INTEGER,
                retail_disc REAL,
                trans_time INTEGER,
                week_no INTEGER,
                coupon_disc REAL,
                coupon_match_disc REAL
            )
            ''')
        # Recreate table for causal_data
        elif table_name == 'causal_data':
            conn.execute('''
            CREATE TABLE IF NOT EXISTS causal_data (
                product_id INTEGER,
                store_id INTEGER,
                week_no INTEGER,
                display TEXT,
                mailer TEXT
            )
            ''')
        
        # Import in chunks to handle large files
        for chunk in pd.read_csv(csv_file, chunksize=chunksize):
            chunk.to_sql(table_name, conn, if_exists='append', index=False)
            total_rows += len(chunk)
            print(f"Imported {total_rows} rows so far...")
        
        end_time = time.time()
        print(f"Imported {total_rows} rows into {table_name} in {end_time - start_time:.2f} seconds.")
    
    # Create primary keys and indexes after import for better performance
    print("Creating primary keys and indexes...")
    
    try:
        # Add primary key to transaction_data
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS pk_transaction_data ON transaction_data(basket_id, household_key, product_id)")
        
        # For causal_data, create a non-unique index instead of a unique index
        # This is because we've observed that there are duplicate combinations of product_id, store_id, and week_no
        conn.execute("CREATE INDEX IF NOT EXISTS idx_causal_data ON causal_data(product_id, store_id, week_no)")
        
        # Alternatively, uncomment the following to deduplicate causal_data before creating a unique index
        # print("Deduplicating causal_data table...")
        # conn.execute("""
        # CREATE TABLE causal_data_deduplicated AS
        # SELECT DISTINCT product_id, store_id, week_no, display, mailer
        # FROM causal_data
        # """)
        # conn.execute("DROP TABLE causal_data")
        # conn.execute("ALTER TABLE causal_data_deduplicated RENAME TO causal_data")
        # conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS pk_causal_data ON causal_data(product_id, store_id, week_no)")
        
        # Create indexes for transaction_data
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transaction_household ON transaction_data(household_key)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transaction_product ON transaction_data(product_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transaction_week ON transaction_data(week_no)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transaction_day ON transaction_data(day)")
    except sqlite3.OperationalError as e:
        print(f"Note: {e}")
    
    # Close the connection
    conn.close()
    
    print("Data import completed.")

def validate_import():
    """Validate the data import by counting rows in each table"""
    
    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tables to check
    tables = [
        'household_demographics',
        'product',
        'campaign_desc',
        'campaign_table',
        'coupon',
        'coupon_redemption',
        'transaction_data',
        'causal_data'
    ]
    
    print("\nValidating data import...")
    
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table}: {count} rows")
    
    # Close the connection
    conn.close()

def find_duplicates():
    """Find duplicate entries in the causal_data table"""
    print("Checking for duplicate entries in causal_data table...")
    
    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Query to find duplicates
    query = """
    SELECT product_id, store_id, week_no, COUNT(*) as count
    FROM causal_data
    GROUP BY product_id, store_id, week_no
    HAVING COUNT(*) > 1
    ORDER BY COUNT(*) DESC
    LIMIT 10
    """
    
    cursor.execute(query)
    duplicates = cursor.fetchall()
    
    if duplicates:
        print("Found duplicate entries in causal_data table:")
        print("product_id | store_id | week_no | count")
        print("-" * 40)
        for row in duplicates:
            print(f"{row[0]:10} | {row[1]:8} | {row[2]:7} | {row[3]}")
            
        # Get total number of duplicates
        cursor.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT product_id, store_id, week_no
            FROM causal_data
            GROUP BY product_id, store_id, week_no
            HAVING COUNT(*) > 1
        )
        """)
        total_duplicate_groups = cursor.fetchone()[0]
        
        # Calculate total duplicate rows using a different approach
        cursor.execute("""
        SELECT SUM(count - 1)
        FROM (
            SELECT COUNT(*) as count
            FROM causal_data
            GROUP BY product_id, store_id, week_no
            HAVING COUNT(*) > 1
        )
        """)
        total_duplicate_rows = cursor.fetchone()[0] or 0
        
        print(f"\nTotal duplicate groups: {total_duplicate_groups}")
        print(f"Total duplicate rows: {total_duplicate_rows}")
    else:
        print("No duplicates found in causal_data table.")
    
    # Close the connection
    conn.close()
    
    return duplicates

def main():
    """Main function to set up the SQLite database"""
    
    # Check if DB already exists
    if os.path.exists(DB_PATH):
        response = input(f"Database {DB_PATH} already exists. Do you want to recreate it? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without making changes.")
            return
        
    # Create database schema
    create_database_schema()
    
    # Import data from CSV files
    import_data()
    
    # Validate the import
    validate_import()
    
    # Check for duplicates in causal_data
    find_duplicates()
    
    print(f"\nDatabase setup complete. Database file: {DB_PATH}")
    print("You can now use this database for your retail analytics project.")

if __name__ == "__main__":
    main() 