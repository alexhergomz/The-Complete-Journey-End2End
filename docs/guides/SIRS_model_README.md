# SIRS Customer Lifecycle Model and Optimization

This module implements a SIRS-like (Susceptible-Infected-Recovered-Susceptible) model for customer lifecycle analysis and optimization.

## Model Overview

The SIRS model adapts epidemiological modeling techniques to customer behavior analysis. In our implementation:

- **Premium** (analogous to Susceptible): Customers who are highly engaged and high-value
- **Value** (analogous to Infected): Customers who are engaged but at moderate value
- **Inactive** (analogous to Recovered): Customers who have minimal engagement

Unlike traditional marketing RFM (Recency, Frequency, Monetary) segmentation, the SIRS approach models the transitions between states over time and allows for optimization of marketing interventions.

## Key Components

### 1. Customer SIRS Model (`customer_sirs_model.py`)

This module:
- Extracts empirical transition probabilities from transaction data
- Calculates state metrics (average revenue, duration)
- Analyzes marketing intervention effects on transitions
- Visualizes customer state evolution over time

### 2. Optimization Engine (`optimize_sirs_interventions.py`)

This module:
- Uses the empirical parameters from the SIRS model
- Simulates customer flows between states with and without interventions
- Optimizes intervention timing to maximize revenue
- Compares targeted strategies for different customer states

## How to Use

### 1. Run the SIRS Model to calculate empirical parameters:

```python
from customer_sirs_model import CustomerSIRSModel

model = CustomerSIRSModel()
results = model.run_analysis()
```

### 2. Run the Optimization:

```python
python optimize_sirs_interventions.py
```

This will:
1. Run a generic optimization to find the best intervention schedule
2. Run targeted optimizations for each customer state
3. Compare and visualize the results

## Optimization Approach

The optimization uses:

1. **Mathematical Simulation**: Uses the empirical transition matrix to simulate customer flows with and without interventions
2. **Sequential Least Squares Programming (SLSQP)**: A constrained optimization algorithm that determines the optimal periods for intervention
3. **Budget Constraints**: Considers the cost of interventions and limits the total number of interventions
4. **Targeted Strategies**: Tests different intervention approaches targeting specific customer states

## Output Files

Results are saved to the `sirs_optimization_results` directory:
- `optimization_results.png`: Visualizations comparing baseline vs. optimized scenarios
- `intervention_schedule.png`: Visual representation of the optimal intervention schedule
- `target_strategy_comparison.png`: Comparison of different targeting strategies
- `optimization_results.json`: Complete results in JSON format for further analysis

## Interpretation

The optimization results provide actionable insights:

1. **Optimal Intervention Schedule**: Identifies the best periods to deploy marketing campaigns
2. **Revenue Improvement**: Quantifies the expected revenue increase from optimized interventions
3. **Target Identification**: Determines which customer state (Premium, Value, or Inactive) should be prioritized
4. **Budget Allocation**: Helps allocate marketing budget more efficiently based on expected returns

## Benefits Over Traditional Methods

- **Dynamic Analysis**: Models customer movement between states over time
- **Intervention Optimization**: Scientifically determines when to deploy marketing efforts
- **Targeted Approach**: Identifies which customer groups provide the highest ROI for interventions
- **Cost Awareness**: Explicitly considers intervention costs in the optimization 