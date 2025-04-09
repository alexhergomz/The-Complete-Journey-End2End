# Retail Analytics Solution: The Complete Journey

This project provides comprehensive retail analytics capabilities using the Dunnhumby "The Complete Journey" dataset. The solution includes customer segmentation, demand forecasting, price optimization, and inventory management analytics.

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

## Project Structure

The project has been organized into the following structure:

```
├── src/                    # Source code
│   ├── data/               # Data processing scripts
│   ├── models/             # Model implementation files
│   ├── utilities/          # Utility functions
│   ├── visualization/      # Visualization scripts
│   ├── notebooks/          # Jupyter notebooks
│   └── database/           # Database scripts
│
├── data/                   # Data storage
│   ├── raw/                # Raw dataset files
│   └── processed/          # Processed dataset files
│
├── docs/                   # Documentation
│   ├── guides/             # Detailed guides
│   └── images/             # Documentation images
│
├── results/                # Analysis and model results
│   ├── exploration/        # EDA results
│   ├── segmentation/       # Customer segmentation results
│   ├── optimization/       # Price optimization results
│   ├── simulation/         # Simulation results
│   └── sirs_model/         # SIRS model results
│
├── requirements.txt        # Python dependencies
└── venv/                   # Virtual environment
```

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
   python src/database/create_sqlite_db.py
   ```
   This script will:
   - Create the SQLite database file
   - Set up the database schema
   - Import the data from CSV files
   - Verify the installation

### 3. Exploring the Data

To run the initial data exploration:

```
python src/data/explore_data.py
```

This will generate several reports and visualizations in the `results/exploration` folder.

Alternatively, you can use the Jupyter notebooks in the `src/notebooks` directory for interactive exploration.

## Project Status

### Completed
- [x] Database setup and schema creation
- [x] Initial data exploration and visualization
- [x] Customer segmentation analysis
- [x] RFM analysis and customer lifetime value modeling
- [x] Time series analysis for demand forecasting
- [x] Price elasticity measurement
- [x] SIRS model implementation for customer behavior

### In Progress
- [ ] Advanced demand forecasting models
- [ ] Interactive dashboard development
- [ ] API development for model integration

### Planned
- [ ] Recommendation engine for cross-selling
- [ ] Basket analysis and association rules
- [ ] Churn prediction and prevention strategies
- [ ] Multi-touch attribution modeling
- [ ] Real-time analytics capabilities

## Components

The project contains several key components:

1. **Customer Analytics**: RFM analysis and customer lifetime value modeling
2. **Demand Forecasting**: Time series analysis and advanced forecasting
3. **Price Optimization**: Price elasticity analysis and optimization models
4. **Inventory Optimization**: Sales velocity analysis and inventory recommendations
5. **SIRS Model**: Susceptible-Infected-Recovered-Susceptible model for customer behavior

## Documentation

Detailed documentation can be found in the `docs` directory:
- General project documentation in `docs/README.md`
- SIRS model documentation in `docs/guides/SIRS_model_README.md`
- Optimization documentation in `docs/guides/optimization_README.md`

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dunnhumby for providing "The Complete Journey" dataset
- The open-source community for the tools and libraries used in this project 