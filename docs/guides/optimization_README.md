# Segment-Specific Variable Optimization

This module implements machine learning-based optimization for key customer behavior variables by segment, using the insights gained from feature importance analysis.

## Overview

The optimization process uses a combination of:
1. **Gradient Boosting Regression** to model the relationship between key variables and segment revenue
2. **Bayesian Optimization** to find optimal parameter values that maximize segment revenue
3. **Segment-specific targeting** of variables identified as significant in our statistical analysis

## Key Variables Optimized

### Value Shoppers (Segment 0)
- **Top Department Spend** - Affects base spending rate
- **Total Items** - Affects intervention effectiveness
- **Unique Products** - Affects response rates to marketing
- **Basket Frequency** - Affects transition probabilities and retention
- **Average Basket Value** - Affects value multipliers

### Premium Shoppers (Segment 1)
- **Top Department Spend** - Affects base spending rate
- **Unique Products** - Affects response rates to marketing
- **Campaign Participation** - Affects intervention effectiveness
- **Active Weeks** - Affects transition probabilities and retention
- **Total Baskets** - Affects value multipliers

## How to Run

```bash
# Install requirements
pip install -r requirements.txt

# Run segment-specific optimization
python optimize_segments.py
```

## Outputs

The optimization process generates:

1. **segment_optimization_results.csv** - CSV file with optimal values and actionable recommendations
2. **segment_parameter_comparison.png** - Visual comparison of parameters common to both segments
3. **segment_specific_parameters.png** - Visual breakdown of all parameters by segment
4. **segment_0_optimization.png** - Revenue comparison for Value Shoppers (baseline vs. optimized)
5. **segment_1_optimization.png** - Revenue comparison for Premium Shoppers (baseline vs. optimized)

## How It Works

1. The optimizer runs multiple simulations with different parameter values
2. Trains a machine learning model to predict segment revenue based on parameters
3. Uses Bayesian optimization to find optimal parameter values
4. Verifies the optimized parameters with a final simulation
5. Provides actionable recommendations based on the optimal values

## Interpretation

The multiplier values in the results can be interpreted as:
- Values > 1.0: Variable should be increased (e.g., 1.2 = increase by 20%)
- Values < 1.0: Variable should be decreased (e.g., 0.8 = decrease by 20%)
- Values ≈ 1.0: Variable is already at optimal level

The actionable recommendations translate these multipliers into specific marketing and operational strategies. 