#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Customer SIRS-like Model (Premium, Value, Inactive states)
Calculates empirical parameters for state transitions using the Complete Journey dataset
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime, timedelta
from tqdm import tqdm
from collections import Counter
from scipy import stats
import json

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'dunnhumby.db')
RESULTS_DIR = os.path.join(BASE_DIR, 'sirs_model_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

class CustomerSIRSModel:
    """
    Customer SIRS (Susceptible-Infected-Recovered-Susceptible) Model
    
    States:
    - Premium (P): High-value, engaged customers (analogous to Susceptible)
    - Value (V): Medium-value, engaged customers (analogous to Infected)
    - Inactive (I): Low-engagement customers (analogous to Recovered)
    
    This model analyzes customer transitions between states and enables 
    targeted marketing intervention optimization.
    """
    
    def __init__(self, data_path="archive/transaction_data.csv"):
        """
        Initialize the Customer SIRS Model
        
        Parameters:
        -----------
        data_path : str
            Path to transaction data CSV
        """
        self.data_path = data_path
        self.data = None
        self.customer_states = None
        self.transition_matrix = None
        self.state_revenues = None
        self.state_durations = None
        self.intervention_effects = None
        self.results_dir = "sirs_model_results"
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_data(self):
        """Load transaction data and prepare for analysis"""
        try:
            self.data = pd.read_csv(self.data_path)
            
            # Sort by customer and date
            self.data = self.data.sort_values(["household_key", "DAY"])
            
            print(f"Loaded {len(self.data)} transactions for {self.data['household_key'].nunique()} customers")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def define_customer_states(self, lookback_period=30, 
                               premium_threshold=100, 
                               value_threshold=25):
        """
        Define customer states based on recency and monetary value
        
        Parameters:
        -----------
        lookback_period : int
            Number of days to look back for determining state
        premium_threshold : float
            Spending threshold for Premium customers
        value_threshold : float
            Spending threshold for Value customers
        """
        if self.data is None:
            self.load_data()
        
        # Get unique days in ascending order
        analysis_days = sorted(self.data["DAY"].unique())
        
        # Create DataFrame to store customer states over time
        customers = self.data["household_key"].unique()
        customer_states = pd.DataFrame(index=customers, columns=analysis_days)
        
        # For each analysis day, determine customer state
        for day in analysis_days:
            # Define the lookback period
            start_day = max(1, day - lookback_period)
            
            # Filter transactions in the lookback period
            period_data = self.data[(self.data["DAY"] >= start_day) & 
                                    (self.data["DAY"] <= day)]
            
            # Calculate customer metrics in the period
            period_metrics = period_data.groupby("household_key").agg(
                total_spend=("SALES_VALUE", "sum"),
                transaction_count=("BASKET_ID", "count"),
                last_day=("DAY", "max")
            ).reset_index()
            
            # Calculate recency
            period_metrics["recency"] = day - period_metrics["last_day"]
            
            # Determine customer state based on spend and recency
            for _, row in period_metrics.iterrows():
                customer_id = row["household_key"]
                
                # Determine state based on spending and recency
                if row["total_spend"] >= premium_threshold and row["recency"] <= 10:
                    customer_states.loc[customer_id, day] = "Premium"
                elif row["total_spend"] >= value_threshold and row["recency"] <= 20:
                    customer_states.loc[customer_id, day] = "Value"
                else:
                    customer_states.loc[customer_id, day] = "Inactive"
        
        # For customers with no transactions in the period, mark as Inactive
        customer_states = customer_states.fillna("Inactive")
        
        self.customer_states = customer_states
        print(f"Defined customer states across {len(analysis_days)} time periods")
        return customer_states
    
    def calculate_transition_matrix(self):
        """
        Calculate the transition probability matrix between customer states
        """
        if self.customer_states is None:
            self.define_customer_states()
        
        # Get unique dates
        dates = self.customer_states.columns
        
        # Initialize transition count matrix
        states = ["Premium", "Value", "Inactive"]
        transition_counts = pd.DataFrame(0, index=states, columns=states)
        
        # Count transitions
        for i in range(len(dates) - 1):
            current_date = dates[i]
            next_date = dates[i+1]
            
            # Get state pairs
            current_states = self.customer_states[current_date]
            next_states = self.customer_states[next_date]
            
            # Count transitions for each customer
            for customer in self.customer_states.index:
                current_state = current_states.loc[customer]
                next_state = next_states.loc[customer]
                
                # Increment transition count
                if pd.notna(current_state) and pd.notna(next_state):
                    transition_counts.loc[current_state, next_state] += 1
        
        # Calculate transition probabilities
        transition_matrix = transition_counts.copy()
        for state in states:
            state_total = transition_counts.loc[state].sum()
            if state_total > 0:
                transition_matrix.loc[state] = transition_counts.loc[state] / state_total
        
        self.transition_matrix = transition_matrix
        print(f"Calculated transition matrix:\n{transition_matrix}")
        return transition_matrix
    
    def calculate_state_metrics(self):
        """
        Calculate revenue and duration metrics for each state
        """
        if self.customer_states is None:
            self.define_customer_states()
        
        if self.data is None:
            self.load_data()
        
        # Initialize state metrics
        states = ["Premium", "Value", "Inactive"]
        state_revenues = {state: 0 for state in states}
        state_counts = {state: 0 for state in states}
        state_durations = {state: [] for state in states}
        
        # Calculate average revenue per state
        for day in self.customer_states.columns:
            # Filter transactions for this day
            day_transactions = self.data[self.data["DAY"] == day]
            
            # Calculate revenue per customer for this day
            if not day_transactions.empty:
                revenue_by_customer = day_transactions.groupby("household_key")["SALES_VALUE"].sum()
                
                for customer in self.customer_states.index:
                    state = self.customer_states.loc[customer, day]
                    if pd.notna(state):
                        # Add revenue if customer had transactions
                        if customer in revenue_by_customer.index:
                            state_revenues[state] += revenue_by_customer.loc[customer]
                            state_counts[state] += 1
        
        # Calculate average revenue per state
        avg_state_revenues = {state: state_revenues[state]/max(1, state_counts[state]) 
                              for state in states}
        
        # Calculate average duration in each state
        for customer in self.customer_states.index:
            customer_states = self.customer_states.loc[customer]
            
            current_state = None
            current_duration = 0
            
            for day, state in customer_states.items():
                if pd.isna(state):
                    continue
                    
                if state != current_state:
                    # Save previous state duration
                    if current_state is not None and current_duration > 0:
                        state_durations[current_state].append(current_duration)
                    
                    # Reset for new state
                    current_state = state
                    current_duration = 1
                else:
                    current_duration += 1
            
            # Add final state duration
            if current_state is not None and current_duration > 0:
                state_durations[current_state].append(current_duration)
        
        # Calculate average duration per state
        avg_state_durations = {state: np.mean(durations) if durations else 0 
                              for state, durations in state_durations.items()}
        
        self.state_revenues = avg_state_revenues
        self.state_durations = avg_state_durations
        
        print(f"Average revenue per state: {avg_state_revenues}")
        print(f"Average duration per state: {avg_state_durations}")
        
        return avg_state_revenues, avg_state_durations
    
    def estimate_intervention_effects(self, intervention_data=None):
        """
        Estimate effect of marketing interventions on transition probabilities
        
        If intervention_data is not provided, uses synthetic data to estimate effects
        """
        if self.transition_matrix is None:
            self.calculate_transition_matrix()
        
        # If no intervention data is provided, use synthetic effects
        if intervention_data is None:
            # Define synthetic intervention effects (multiplicative)
            # Format: {from_state: {to_state: effect_multiplier}}
            self.intervention_effects = {
                "Premium": {"Premium": 1.2, "Value": 0.7, "Inactive": 0.5},  # Keep customers Premium
                "Value": {"Premium": 1.5, "Value": 1.1, "Inactive": 0.6},    # Try to upgrade to Premium
                "Inactive": {"Premium": 2.0, "Value": 2.0, "Inactive": 0.8}  # Reactivate customers
            }
            
            print("Using synthetic intervention effects:")
            for state, effects in self.intervention_effects.items():
                print(f"  From {state}: {effects}")
            
        else:
            # Real implementation would analyze actual intervention data
            # to calculate effect multipliers
            pass
            
        return self.intervention_effects
    
    def simulate_customer_flow(self, periods=12, initial_distribution=None, 
                              interventions=None, plot=True):
        """
        Simulate customer flow between states over time
        
        Parameters:
        -----------
        periods : int
            Number of periods to simulate
        initial_distribution : dict
            Initial distribution of customers across states
            Format: {"Premium": 0.2, "Value": 0.3, "Inactive": 0.5}
        interventions : list
            List of periods when marketing interventions occur
        plot : bool
            Whether to plot the results
        
        Returns:
        --------
        DataFrame with state distribution over time and revenue
        """
        if self.transition_matrix is None:
            self.calculate_transition_matrix()
            
        if self.state_revenues is None:
            self.calculate_state_metrics()
            
        if self.intervention_effects is None:
            self.estimate_intervention_effects()
            
        # Set default initial distribution if not provided
        if initial_distribution is None:
            # Calculate from data if available
            if self.customer_states is not None and not self.customer_states.empty:
                first_date = self.customer_states.columns[0]
                state_counts = self.customer_states[first_date].value_counts()
                total = state_counts.sum()
                initial_distribution = {
                    "Premium": state_counts.get("Premium", 0) / total,
                    "Value": state_counts.get("Value", 0) / total,
                    "Inactive": state_counts.get("Inactive", 0) / total
                }
            else:
                # Default distribution
                initial_distribution = {"Premium": 0.2, "Value": 0.3, "Inactive": 0.5}
                
        # Initialize simulation dataframe
        states = ["Premium", "Value", "Inactive"]
        sim_data = pd.DataFrame(0, index=range(periods+1), 
                               columns=states + ["Revenue", "Intervention"])
        
        # Set initial distribution
        for state in states:
            sim_data.loc[0, state] = initial_distribution.get(state, 0)
        
        # Calculate initial revenue
        sim_data.loc[0, "Revenue"] = sum(sim_data.loc[0, state] * self.state_revenues[state] 
                                       for state in states)
        
        # Mark interventions
        if interventions is not None:
            for period in interventions:
                if 1 <= period <= periods:
                    sim_data.loc[period, "Intervention"] = 1
        
        # Run simulation
        for period in range(1, periods+1):
            has_intervention = bool(sim_data.loc[period, "Intervention"])
            
            # Calculate new distribution based on transition probabilities
            for to_state in states:
                # Sum transitions from all states to this state
                new_prob = 0
                for from_state in states:
                    base_trans_prob = self.transition_matrix.loc[from_state, to_state]
                    
                    # Apply intervention effect if applicable
                    if has_intervention and self.intervention_effects:
                        effect_multiplier = self.intervention_effects[from_state][to_state]
                        trans_prob = base_trans_prob * effect_multiplier
                        # Ensure probability doesn't exceed 1
                        trans_prob = min(trans_prob, 1.0)
                    else:
                        trans_prob = base_trans_prob
                    
                    new_prob += sim_data.loc[period-1, from_state] * trans_prob
                
                sim_data.loc[period, to_state] = new_prob
            
            # Normalize to ensure probabilities sum to 1
            state_sum = sum(sim_data.loc[period, state] for state in states)
            if state_sum > 0:  # Avoid division by zero
                for state in states:
                    sim_data.loc[period, state] /= state_sum
            
            # Calculate revenue for this period
            sim_data.loc[period, "Revenue"] = sum(sim_data.loc[period, state] * self.state_revenues[state] 
                                               for state in states)
        
        # Calculate cumulative revenue
        sim_data["Cumulative_Revenue"] = sim_data["Revenue"].cumsum()
        
        # Plot results if requested
        if plot:
            self._plot_simulation_results(sim_data, interventions)
        
        return sim_data
    
    def _plot_simulation_results(self, sim_data, interventions=None):
        """Plot simulation results"""
        plt.figure(figsize=(15, 10))
        
        # Plot customer distribution
        plt.subplot(2, 1, 1)
        for state in ["Premium", "Value", "Inactive"]:
            plt.plot(sim_data.index, sim_data[state], 
                    label=state, linewidth=2)
        
        # Mark interventions if provided
        if interventions:
            for period in interventions:
                plt.axvline(x=period, color="r", linestyle="--", alpha=0.5)
        
        plt.title("Customer State Distribution Over Time")
        plt.xlabel("Period")
        plt.ylabel("Proportion of Customers")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot revenue
        plt.subplot(2, 1, 2)
        plt.plot(sim_data.index, sim_data["Revenue"], 
                label="Period Revenue", linewidth=2)
        plt.plot(sim_data.index, sim_data["Cumulative_Revenue"]/10, 
                label="Cumulative Revenue (scaled)", linewidth=2, linestyle="--")
        
        # Mark interventions if provided
        if interventions:
            for period in interventions:
                plt.axvline(x=period, color="r", linestyle="--", alpha=0.5,
                          label="Intervention" if period == interventions[0] else "")
        
        plt.title("Revenue Over Time")
        plt.xlabel("Period")
        plt.ylabel("Revenue")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/simulation_results.png", dpi=300)
        plt.close()
    
    def compare_intervention_strategies(self, strategies, periods=12, initial_distribution=None):
        """
        Compare different intervention strategies
        
        Parameters:
        -----------
        strategies : dict
            Dictionary of strategies to compare
            Format: {"Strategy Name": [intervention_periods]}
        periods : int
            Number of periods to simulate
        initial_distribution : dict
            Initial distribution of customers across states
            
        Returns:
        --------
        Dictionary with simulation results for each strategy
        """
        results = {}
        
        # Run simulation for each strategy
        for name, interventions in strategies.items():
            print(f"Simulating strategy: {name}")
            sim_data = self.simulate_customer_flow(
                periods=periods,
                initial_distribution=initial_distribution,
                interventions=interventions,
                plot=False
            )
            results[name] = {
                "simulation_data": sim_data,
                "total_revenue": sim_data["Revenue"].sum(),
                "final_distribution": {
                    state: sim_data.loc[periods, state] 
                    for state in ["Premium", "Value", "Inactive"]
                },
                "interventions": interventions,
                "intervention_count": len(interventions)
            }
        
        # Plot comparison
        self._plot_strategy_comparison(results, periods)
        
        return results
    
    def _plot_strategy_comparison(self, results, periods):
        """Plot comparison of different strategies"""
        plt.figure(figsize=(15, 10))
        
        # Plot revenue comparison
        plt.subplot(2, 1, 1)
        
        for name, result in results.items():
            sim_data = result["simulation_data"]
            plt.plot(sim_data.index, sim_data["Revenue"], 
                    label=f"{name} (Total: ${result['total_revenue']:.2f})", 
                    linewidth=2)
            
            # Mark interventions
            for period in result["interventions"]:
                plt.axvline(x=period, color=plt.gca().lines[-1].get_color(), 
                          linestyle="--", alpha=0.3)
        
        plt.title("Revenue Comparison Across Strategies")
        plt.xlabel("Period")
        plt.ylabel("Revenue per Period")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot final state distribution
        plt.subplot(2, 1, 2)
        
        strategies = list(results.keys())
        states = ["Premium", "Value", "Inactive"]
        
        state_data = {state: [results[strategy]["final_distribution"][state] 
                             for strategy in strategies] 
                     for state in states}
        
        x = np.arange(len(strategies))
        width = 0.25
        
        for i, state in enumerate(states):
            plt.bar(x + i*width - width, state_data[state], 
                   width=width, label=state)
        
        plt.title("Final Customer Distribution by Strategy")
        plt.xlabel("Strategy")
        plt.ylabel("Proportion of Customers")
        plt.xticks(x, strategies)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/strategy_comparison.png", dpi=300)
        plt.close()
    
    def run_analysis(self):
        """
        Run the full SIRS analysis pipeline
        
        Returns:
        --------
        Dictionary with all analysis results
        """
        print("Starting SIRS model analysis...")
        
        # Load and prepare data
        self.load_data()
        
        # Define customer states
        # Use a smaller lookback and adjust thresholds for this dataset
        self.define_customer_states(lookback_period=15, premium_threshold=50, value_threshold=15)
        
        # Calculate transition matrix
        self.calculate_transition_matrix()
        
        # Calculate state metrics
        self.calculate_state_metrics()
        
        # Estimate intervention effects
        self.estimate_intervention_effects()
        
        # Run baseline simulation (no interventions)
        baseline_sim = self.simulate_customer_flow(interventions=None)
        
        # Run simulation with regular interventions (every 3 periods)
        regular_interventions = list(range(3, 13, 3))  # periods 3, 6, 9, 12
        regular_sim = self.simulate_customer_flow(interventions=regular_interventions)
        
        # Compare intervention strategies
        strategies = {
            "No Interventions": [],
            "Regular (Quarterly)": regular_interventions,
            "Front-loaded": [1, 2, 3, 4],
            "Back-loaded": [9, 10, 11, 12]
        }
        strategy_results = self.compare_intervention_strategies(strategies)
        
        # Compile all results
        analysis_results = {
            "transition_matrix": self.transition_matrix.to_dict(),
            "state_revenues": self.state_revenues,
            "state_durations": self.state_durations,
            "intervention_effects": self.intervention_effects,
            "baseline_simulation": baseline_sim.to_dict(),
            "regular_intervention_simulation": regular_sim.to_dict(),
            "strategy_comparison": {
                name: {k: v for k, v in data.items() if k != "simulation_data"} 
                for name, data in strategy_results.items()
            }
        }
        
        # Save results
        with open(f"{self.results_dir}/analysis_results.json", "w") as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"Analysis complete. Results saved to {self.results_dir}/")
        return analysis_results

def main():
    print("=" * 60)
    print("Customer SIRS Model Analysis")
    print("Analyzing customer transitions between Premium, Value, and Inactive states")
    print("=" * 60)
    
    model = CustomerSIRSModel()
    results = model.run_analysis()
    
    print("\nKey findings:")
    print(f"- Transition probabilities: {'Calculated' if model.transition_matrix is not None else 'Failed'}")
    print(f"- State revenues: {results.get('state_revenues', 'Not calculated')}")
    print(f"- Intervention effects: {'Estimated' if model.intervention_effects is not None else 'Failed'}")
    print("\nCheck the sirs_model_results directory for detailed output and visualizations.")

if __name__ == "__main__":
    main() 