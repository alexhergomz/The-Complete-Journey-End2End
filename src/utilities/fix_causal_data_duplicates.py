import os
import sqlite3
import time
import pandas as pd

# Path to the SQLite database
DB_PATH = "dunnhumby.db"

def check_duplicates():
    """Check for duplicate entries in the causal_data table"""
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

def deduplicate_causal_data():
    """Deduplicate the causal_data table by keeping only unique combinations"""
    print("Deduplicating causal_data table...")
    
    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get current row count
    cursor.execute("SELECT COUNT(*) FROM causal_data")
    original_count = cursor.fetchone()[0]
    print(f"Original row count: {original_count}")
    
    # Start timing
    start_time = time.time()
    
    # Create a new table with deduplicated data
    print("Creating temporary table with deduplicated data...")
    cursor.execute("""
    CREATE TABLE causal_data_deduplicated AS
    SELECT DISTINCT product_id, store_id, week_no, display, mailer
    FROM causal_data
    """)
    
    # Get new row count
    cursor.execute("SELECT COUNT(*) FROM causal_data_deduplicated")
    new_count = cursor.fetchone()[0]
    print(f"New row count: {new_count}")
    print(f"Removed {original_count - new_count} duplicate rows.")
    
    # Replace the original table with the deduplicated one
    print("Replacing original table with deduplicated data...")
    cursor.execute("DROP TABLE causal_data")
    cursor.execute("ALTER TABLE causal_data_deduplicated RENAME TO causal_data")
    
    # Create a unique index on the deduplicated table
    print("Creating unique index on deduplicated table...")
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS pk_causal_data ON causal_data(product_id, store_id, week_no)")
    
    # Commit changes
    conn.commit()
    
    end_time = time.time()
    print(f"Deduplication completed in {end_time - start_time:.2f} seconds.")
    
    # Close the connection
    conn.close()

def create_non_unique_index():
    """Create a non-unique index instead of a unique index for causal_data"""
    print("Creating non-unique index for causal_data...")
    
    # Connect to the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Try to drop the unique index if it exists
    try:
        cursor.execute("DROP INDEX IF EXISTS pk_causal_data")
    except sqlite3.OperationalError as e:
        print(f"Note: {e}")
    
    # Create a non-unique index
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_causal_data ON causal_data(product_id, store_id, week_no)")
    
    # Commit changes
    conn.commit()
    print("Non-unique index created successfully.")
    
    # Close the connection
    conn.close()

def main():
    """Main function to check and fix duplicates in causal_data"""
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file {DB_PATH} not found.")
        return
    
    # Check for duplicates first
    duplicates = check_duplicates()
    
    if duplicates:
        print("\nChoose an option to fix duplicates:")
        print("1. Deduplicate data (keep only unique combinations)")
        print("2. Use non-unique index instead of unique index")
        print("3. Exit without making changes")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == "1":
            deduplicate_causal_data()
            # Verify deduplication
            check_duplicates()
        elif choice == "2":
            create_non_unique_index()
        else:
            print("Exiting without making changes.")
    else:
        print("No duplicates found. No action needed.")

if __name__ == "__main__":
    main() 