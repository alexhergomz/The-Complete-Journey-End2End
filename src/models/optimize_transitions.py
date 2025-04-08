import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import json
import os

# Create results directory if it doesn't exist
os.makedirs('optimization_results', exist_ok=True)

def main():
    print("=" * 60)
    print("Transition Matrix Optimization for Maximum LTV")
    print("=" * 60)
    
    # 1. Get transition matrix from SIRS model analysis results
    try:
        with open("sirs_model_results/analysis_results.json", "r") as f:
            sirs_results = json.load(f)
            transition_matrix = sirs_results["transition_matrix"]
            state_revenues = sirs_results["state_revenues"]
    except FileNotFoundError:
        print("SIRS model results not found. Using default values...")
        # Current transition matrix from SIRS model
        transition_matrix = {
            "Premium": {"Premium": 0.960, "Value": 0.038, "Inactive": 0.002},
            "Value": {"Premium": 0.066, "Value": 0.876, "Inactive": 0.059},
            "Inactive": {"Premium": 0.009, "Value": 0.019, "Inactive": 0.971}
        }
        # State revenue values
        state_revenues = {
            "Premium": 41.57,
            "Value": 15.61,
            "Inactive": 6.50
        }
    
    # Convert dictionaries to numpy arrays for optimization
    states = ["Premium", "Value", "Inactive"]
    P_current = np.zeros((3, 3))
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            P_current[i, j] = transition_matrix[from_state][to_state]
    
    revenue_vector = np.array([state_revenues[state] for state in states])
    
    print("\nCurrent Transition Matrix:")
    print_matrix(P_current, states)
    
    print("\nState Revenues:")
    for i, state in enumerate(states):
        print(f"{state}: ${revenue_vector[i]:.2f}")
    
    # Initial state distribution (based on SIRS model after 12 periods)
    # This can be obtained from the baseline_simulation in sirs_results
    if "baseline_simulation" in sirs_results:
        try:
            initial_distribution = np.array([
                float(sirs_results["baseline_simulation"]["Premium"]["12"]),
                float(sirs_results["baseline_simulation"]["Value"]["12"]),
                float(sirs_results["baseline_simulation"]["Inactive"]["12"])
            ])
        except (KeyError, ValueError):
            initial_distribution = np.array([0.13, 0.12, 0.75])
    else:
        initial_distribution = np.array([0.13, 0.12, 0.75])
    
    print("\nInitial State Distribution:")
    for i, state in enumerate(states):
        print(f"{state}: {initial_distribution[i]:.4f}")
    
    # Current LTV (calculated using current transition matrix)
    current_ltv = -calculate_ltv(P_current.flatten(), revenue_vector, initial_distribution)
    print(f"\nCurrent LTV (36 periods): ${current_ltv:.2f}")
    
    # Run optimization
    print("\nOptimizing transition matrix for maximum LTV...")
    
    # Define bounds on transition probabilities
    # Bounded by current values ±30 percentage points, but within [0,1]
    lower_bounds = np.maximum(0, P_current.flatten() - 0.3)
    upper_bounds = np.minimum(1, P_current.flatten() + 0.3)
    bounds = list(zip(lower_bounds, upper_bounds))
    
    # Initial guess (starting with current matrix)
    x0 = P_current.flatten()
    
    # Run the optimization
    constraint = {'type': 'eq', 'fun': row_sum_constraint}
    result = optimize.minimize(
        calculate_ltv,
        x0,
        args=(revenue_vector, initial_distribution),
        method='SLSQP',
        bounds=bounds,
        constraints=constraint,
        options={'disp': True, 'maxiter': 1000}
    )
    
    # Check if optimization was successful
    if result.success:
        print("\nOptimization successful!")
    else:
        print("\nWarning: Optimization may not have converged.")
        print(f"Status: {result.message}")
    
    # Reshape result to transition matrix
    P_optimal = result.x.reshape(3, 3)
    
    # Normalize rows to ensure they sum exactly to 1 (due to potential numerical errors)
    for i in range(P_optimal.shape[0]):
        P_optimal[i, :] = P_optimal[i, :] / P_optimal[i, :].sum()
    
    print("\nOptimal Transition Matrix:")
    print_matrix(P_optimal, states)
    
    # Calculate optimized LTV
    optimized_ltv = -calculate_ltv(P_optimal.flatten(), revenue_vector, initial_distribution)
    print(f"\nOptimized LTV (36 periods): ${optimized_ltv:.2f}")
    print(f"LTV Improvement: ${optimized_ltv - current_ltv:.2f} (+{(optimized_ltv/current_ltv - 1)*100:.1f}%)")
    
    # Calculate long-term steady state
    steady_state_current = calculate_steady_state(P_current)
    steady_state_optimal = calculate_steady_state(P_optimal)
    
    print("\nSteady State Distribution Comparison:")
    print("State      | Current  | Optimal  | Change")
    print("-----------|----------|----------|--------")
    for i, state in enumerate(states):
        print(f"{state:10} | {steady_state_current[i]:.6f} | {steady_state_optimal[i]:.6f} | {steady_state_optimal[i] - steady_state_current[i]:+.6f}")
    
    # Calculate key transition improvements
    print("\nKey Transition Improvements:")
    print("Transition         | Current  | Optimal  | Change")
    print("-------------------|----------|----------|--------")
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            if i != j or from_state == "Premium":  # Only show non-diagonal transitions and Premium retention
                print(f"{from_state:8} → {to_state:8} | {P_current[i, j]:.6f} | {P_optimal[i, j]:.6f} | {P_optimal[i, j] - P_current[i, j]:+.6f}")
    
    # Plot comparison of current vs optimal distributions over time
    plot_distribution_comparison(P_current, P_optimal, initial_distribution, states, revenue_vector, periods=36)
    
    # Create a detailed report of results
    create_optimization_report(P_current, P_optimal, states, current_ltv, optimized_ltv, 
                               steady_state_current, steady_state_optimal, revenue_vector)
    
    print("\nOptimization complete! Results saved to optimization_results/")


def calculate_ltv(P_flat, revenue_vector, initial_distribution, discount_factor=0.95, periods=36):
    """
    Calculate discounted LTV using a transition matrix
    
    Parameters:
    -----------
    P_flat : flattened transition matrix (9 elements)
    revenue_vector : average revenue for each state
    initial_distribution : starting state distribution
    discount_factor : discount rate for future revenues
    periods : number of periods to calculate
    
    Returns:
    --------
    Negative LTV (for minimization)
    """
    # Reshape flat array to 3x3 matrix
    P = P_flat.reshape(3, 3)
    
    # Initialize with current distribution
    state_dist = initial_distribution.copy()
    total_ltv = 0
    
    # Calculate LTV over multiple periods
    for t in range(periods):
        # Revenue in current period
        period_revenue = np.dot(state_dist, revenue_vector)
        
        # Discount by time
        discounted_revenue = period_revenue * (discount_factor ** t)
        total_ltv += discounted_revenue
        
        # Update state distribution for next period
        state_dist = np.dot(state_dist, P)
    
    # Return negative value for minimization
    return -total_ltv


def row_sum_constraint(P_flat):
    """
    Constraint function ensuring each row of the transition matrix sums to 1
    """
    P = P_flat.reshape(3, 3)
    return np.array([
        P[0, 0] + P[0, 1] + P[0, 2] - 1,
        P[1, 0] + P[1, 1] + P[1, 2] - 1,
        P[2, 0] + P[2, 1] + P[2, 2] - 1
    ])


def calculate_steady_state(P, max_iter=1000, tol=1e-8):
    """
    Calculate the steady state distribution of a transition matrix
    using power iteration method
    """
    n = P.shape[0]
    
    # Start with uniform distribution
    v = np.ones(n) / n
    
    # Power iteration
    for _ in range(max_iter):
        v_new = np.dot(v, P)
        
        # Check convergence
        if np.max(np.abs(v_new - v)) < tol:
            return v_new
        
        v = v_new
    
    return v


def print_matrix(matrix, labels):
    """
    Pretty-print a matrix with row and column labels
    """
    n = len(labels)
    
    print(" " * 10, end="")
    for j in range(n):
        print(f"{labels[j]:>10}", end="")
    print()
    
    for i in range(n):
        print(f"{labels[i]:<10}", end="")
        for j in range(n):
            print(f"{matrix[i, j]:>10.6f}", end="")
        print()


def plot_distribution_comparison(P_current, P_optimal, initial_distribution, states, revenue_vector, periods=36):
    """
    Plot the state distribution over time for current and optimal transition matrices
    """
    # Calculate distribution evolution
    dist_current = np.zeros((periods, len(states)))
    dist_optimal = np.zeros((periods, len(states)))
    
    state = initial_distribution.copy()
    for t in range(periods):
        dist_current[t] = state
        state = np.dot(state, P_current)
    
    state = initial_distribution.copy()
    for t in range(periods):
        dist_optimal[t] = state
        state = np.dot(state, P_optimal)
    
    # Plot
    plt.figure(figsize=(15, 10))
    
    # Current distribution
    plt.subplot(2, 1, 1)
    for i, state in enumerate(states):
        plt.plot(range(periods), dist_current[:, i], label=state)
    
    plt.title('State Distribution Over Time: Current Transition Matrix')
    plt.xlabel('Period')
    plt.ylabel('Proportion of Customers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Optimal distribution
    plt.subplot(2, 1, 2)
    for i, state in enumerate(states):
        plt.plot(range(periods), dist_optimal[:, i], label=state)
    
    plt.title('State Distribution Over Time: Optimal Transition Matrix')
    plt.xlabel('Period')
    plt.ylabel('Proportion of Customers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimization_results/distribution_comparison.png')
    plt.close()
    
    # Calculate and plot revenue over time
    revenue_current = np.zeros(periods)
    revenue_optimal = np.zeros(periods)
    
    for t in range(periods):
        revenue_current[t] = np.dot(dist_current[t], revenue_vector)
        revenue_optimal[t] = np.dot(dist_optimal[t], revenue_vector)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(periods), revenue_current, label='Current', linestyle='-')
    plt.plot(range(periods), revenue_optimal, label='Optimal', linestyle='-')
    plt.title('Revenue Comparison Over Time')
    plt.xlabel('Period')
    plt.ylabel('Average Revenue per Customer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('optimization_results/revenue_comparison.png')
    plt.close()


def create_optimization_report(P_current, P_optimal, states, current_ltv, optimized_ltv, 
                              steady_state_current, steady_state_optimal, revenue_vector):
    """
    Create a detailed report of optimization results
    """
    report = {
        "current_transition_matrix": {
            states[i]: {states[j]: P_current[i, j] for j in range(len(states))} 
            for i in range(len(states))
        },
        "optimal_transition_matrix": {
            states[i]: {states[j]: P_optimal[i, j] for j in range(len(states))} 
            for i in range(len(states))
        },
        "current_ltv": current_ltv,
        "optimized_ltv": optimized_ltv,
        "ltv_improvement": optimized_ltv - current_ltv,
        "ltv_improvement_percentage": (optimized_ltv/current_ltv - 1) * 100,
        "current_steady_state": {states[i]: steady_state_current[i] for i in range(len(states))},
        "optimal_steady_state": {states[i]: steady_state_optimal[i] for i in range(len(states))},
        "state_revenues": {states[i]: revenue_vector[i] for i in range(len(states))},
        "key_transition_improvements": {
            f"{states[i]}_to_{states[j]}": {
                "current": P_current[i, j],
                "optimal": P_optimal[i, j],
                "change": P_optimal[i, j] - P_current[i, j],
                "percentage_change": (P_optimal[i, j]/max(P_current[i, j], 0.0001) - 1) * 100
            }
            for i in range(len(states)) for j in range(len(states)) if i != j or states[i] == "Premium"
        }
    }
    
    # Save as JSON
    with open('optimization_results/optimization_results.json', 'w') as f:
        json.dump(report, f, indent=4)


if __name__ == "__main__":
    main() 