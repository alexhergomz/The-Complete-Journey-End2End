# Integrated Retail Analytics Platform

This project builds an end-to-end retail analytics platform using the Dunnhumby "The Complete Journey" dataset. The platform provides customer segmentation, demand forecasting, price optimization, and inventory management capabilities.

## Dataset Overview

The Dunnhumby "The Complete Journey" dataset is a comprehensive retail dataset containing household-level transactions over two years from a group of 2,500 households who are frequent shoppers at a retailer. It contains the following files:

- `transaction_data.csv`: Over 2.5 million rows of retail transactions
- `product.csv`: Attributes for all products available for purchase
- `hh_demographic.csv`: Demographic information for ~800 households
- `coupon.csv`: Coupon information for campaigns sent to customers
- `coupon_redempt.csv`: Coupon redemption data
- `campaign_desc.csv`: Campaign descriptions and schedule
- `campaign_table.csv`: Campaign - household table
- `causal_data.csv`: Product, store, and week level details for display and mailer advertisements

## Project Roadmap

This project follows an 8-phase development roadmap:

1. **Data Acquisition and Storage**: SQLite database setup and initial SQL exploration
2. **Exploratory Data Analysis**: Data cleaning and visual exploration
3. **Customer Analytics**: RFM analysis and customer lifetime value modeling
4. **Demand Forecasting**: Time series analysis and advanced forecasting
5. **Price Optimization**: Price elasticity analysis and optimization models
6. **Inventory Optimization**: Sales velocity analysis and inventory recommendations
7. **Integration and Dashboard**: API development and interactive dashboards
8. **Documentation and Presentation**: Technical documentation and business value presentation

## Setup Instructions

### 1. Prerequisites

- Python 3.8+ recommended
- 16GB RAM recommended (the transaction dataset is large)
- At least 2GB of free disk space

### 2. Installation

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up the SQLite database:
   ```
   python create_sqlite_db.py
   ```
   This script will:
   - Create the SQLite database file
   - Set up the database schema
   - Import the data from CSV files
   - Verify the installation

### 3. Exploring the Data

To run the initial data exploration:

```
python explore_sqlite_data.py
```

This will generate several reports and visualizations in the `exploration_results` folder.

Alternatively, you can use the Jupyter notebook for interactive exploration:

```
jupyter notebook retail_analytics_sqlite.ipynb
```

## Project Structure

- `create_sqlite_db.py`: SQLite database setup script
- `explore_sqlite_data.py`: Initial data exploration script
- `retail_analytics_sqlite.ipynb`: Jupyter notebook for interactive exploration
- `requirements.txt`: Python dependencies
- `fix_causal_data_duplicates.py`: Utility script to handle duplicate entries in the causal_data table

## Common Issues and Solutions

### Duplicate Data in Causal_Data Table

The `causal_data.csv` file contains duplicate combinations of `product_id`, `store_id`, and `week_no`. This can cause issues when creating a unique index or primary key on these columns. The project provides two solutions:

1. Use a non-unique index instead (default in the latest version)
2. Deduplicate the data by running:
   ```
   python fix_causal_data_duplicates.py
   ```
   
This utility script will:
- Check for duplicate entries
- Show statistics about duplicates
- Provide options to either:
  - Deduplicate the data (keeping one record per unique combination)
  - Switch to a non-unique index
  - Exit without making changes

## SQLite Database Advantages

We're using SQLite for this project for several reasons:
- No installation required - SQLite is already built into Python
- Self-contained database file (no server setup)
- Portable - the entire database is in a single file
- Reliable and stable - ACID compliant transactions
- Great for data analysis and exploration workflows

## Future Development

The roadmap outlines the full progression of the project. After setting up the database and initial exploration, the next steps will be:

1. Developing the customer segmentation module
2. Building time series forecasting capabilities
3. Creating price optimization models
4. Implementing inventory management recommendations
5. Building the integrated dashboard

## Contributors

- Your Name

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dunnhumby for providing "The Complete Journey" dataset
- The open-source community for the tools and libraries used in this project 