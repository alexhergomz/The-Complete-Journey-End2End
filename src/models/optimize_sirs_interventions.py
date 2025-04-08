#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SIRS Model Intervention Optimization
Optimizes marketing interventions based on empirical SIRS model parameters
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import json
from customer_sirs_model import CustomerSIRSModel
from tqdm import tqdm

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'sirs_optimization_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

class SIRSOptimizer:
    """Optimizer for SIRS model-based marketing interventions"""
    
    def __init__(self, sirs_model_results=None):
        """
        Initialize the optimizer
        
        Parameters:
        - sirs_model_results: Dict with SIRS model parameters (or None to load from a saved model)
        """
        if sirs_model_results:
            self.model_params = sirs_model_results
        else:
            # If no results provided, run the SIRS model
            print("No model results provided. Running SIRS model...")
            model = CustomerSIRSModel()
            self.model_params = model.run_analysis()
            
        # Extract key parameters
        self.transition_matrix = self.model_params['transition_matrix']
        self.state_counts = self.model_params['state_counts']
        self.revenue_by_state = self.model_params['revenue_by_state']
        self.campaign_effects = self.model_params['campaign_effects']
        
        # Calculate average campaign effect matrix
        self.avg_effect_matrix = self._calculate_avg_effect_matrix()
    
    def _calculate_avg_effect_matrix(self):
        """Calculate the average effect matrix from all campaign effects"""
        if not self.campaign_effects:
            print("Warning: No campaign effects data available")
            return np.zeros((3, 3))
        
        avg_effect = np.zeros((3, 3))
        count = 0
        
        for effect in self.campaign_effects:
            avg_effect += effect['effect_matrix']
            count += 1
        
        if count > 0:
            avg_effect /= count
            
        return avg_effect
    
    def simulate_customer_flow(self, initial_state=None, periods=12, intervention_schedule=None):
        """
        Simulate customer flow between states over time
        
        Parameters:
        - initial_state: Initial state distribution [Premium, Value, Inactive]
                        (if None, use empirical distribution)
        - periods: Number of time periods to simulate
        - intervention_schedule: Dict mapping period -> intervention flag (1 or 0)
        
        Returns:
        - state_history: Array of shape (periods+1, 3) with state populations
        - revenue_history: Array of length periods with total revenue
        """
        # Set initial state
        if initial_state is None:
            # Use empirical distribution
            total = sum(self.state_counts.values())
            state_vector = np.array([
                self.state_counts['Premium'] / total,
                self.state_counts['Value'] / total,
                self.state_counts['Inactive'] / total
            ])
        else:
            state_vector = np.array(initial_state)
            
        # Normalize initial state
        state_vector = state_vector / state_vector.sum()
        
        # Initialize history
        state_history = np.zeros((periods + 1, 3))
        state_history[0] = state_vector
        
        revenue_history = np.zeros(periods)
        
        # Default to no interventions
        if intervention_schedule is None:
            intervention_schedule = {i: 0 for i in range(periods)}
        
        # Run simulation
        for t in range(periods):
            # Check if intervention applied this period
            intervention = bool(intervention_schedule.get(t, 0))
            
            # Select transition matrix based on intervention
            if intervention:
                # Apply the effect matrix to the base transition matrix
                trans_matrix = self.transition_matrix + self.avg_effect_matrix
                # Ensure valid probabilities
                trans_matrix = np.clip(trans_matrix, 0, 1)
                # Normalize rows to sum to 1
                row_sums = trans_matrix.sum(axis=1)
                trans_matrix = trans_matrix / row_sums[:, np.newaxis]
            else:
                trans_matrix = self.transition_matrix
            
            # Calculate next state
            next_state = np.dot(state_vector, trans_matrix)
            
            # Update state history
            state_history[t+1] = next_state
            
            # Calculate revenue for this period
            revenue = (
                next_state[0] * self.revenue_by_state['Premium'] + 
                next_state[1] * self.revenue_by_state['Value'] + 
                next_state[2] * self.revenue_by_state['Inactive']
            )
            revenue_history[t] = revenue
            
            # Update state for next period
            state_vector = next_state
        
        return state_history, revenue_history
    
    def optimize_interventions(self, periods=12, population_size=10000, budget_constraint=0.3, 
                              intervention_cost=0.1):
        """
        Optimize the intervention schedule to maximize revenue within budget constraint
        
        Parameters:
        - periods: Number of time periods to simulate
        - population_size: Number of customers to simulate
        - budget_constraint: Maximum fraction of periods with interventions (0.0-1.0)
        - intervention_cost: Cost of intervention per customer per period
        
        Returns:
        - optimal_schedule: Dict with optimal intervention schedule
        - baseline_results: Dict with baseline scenario results
        - optimized_results: Dict with optimized scenario results
        """
        print("Optimizing intervention schedule...")
        
        # Define objective function for optimization (negative total revenue)
        def objective_function(x):
            # Convert binary array to intervention schedule
            schedule = {t: int(x[t]) for t in range(periods)}
            
            # Run simulation with this schedule
            _, revenue_history = self.simulate_customer_flow(periods=periods, intervention_schedule=schedule)
            
            # Calculate total cost of interventions
            intervention_periods = sum(schedule.values())
            total_cost = intervention_periods * intervention_cost * population_size
            
            # Calculate net revenue (revenue - cost)
            net_revenue = np.sum(revenue_history) * population_size - total_cost
            
            # Check budget constraint (maximum number of interventions)
            max_interventions = int(periods * budget_constraint)
            if intervention_periods > max_interventions:
                # Apply penalty for exceeding budget
                penalty = 1e6 * (intervention_periods - max_interventions)
                net_revenue -= penalty
            
            # Return negative revenue (since we're minimizing)
            return -net_revenue
        
        # Initial guess (no interventions)
        initial_guess = np.zeros(periods)
        
        # Bounds (binary 0/1 decisions)
        bounds = [(0, 1) for _ in range(periods)]
        
        # Run optimization
        result = minimize(
            objective_function,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            options={'ftol': 1e-6, 'disp': True, 'maxiter': 100}
        )
        
        # Convert result to intervention schedule (round to binary 0/1)
        optimized_schedule = {t: round(result.x[t]) for t in range(periods)}
        
        # Run baseline (no interventions) for comparison
        baseline_schedule = {t: 0 for t in range(periods)}
        baseline_state_history, baseline_revenue_history = self.simulate_customer_flow(
            periods=periods, intervention_schedule=baseline_schedule
        )
        
        # Run optimized schedule
        optimized_state_history, optimized_revenue_history = self.simulate_customer_flow(
            periods=periods, intervention_schedule=optimized_schedule
        )
        
        # Calculate metrics
        baseline_total_revenue = np.sum(baseline_revenue_history) * population_size
        optimized_total_revenue = np.sum(optimized_revenue_history) * population_size
        
        intervention_count = sum(optimized_schedule.values())
        intervention_cost_total = intervention_count * intervention_cost * population_size
        
        optimized_net_revenue = optimized_total_revenue - intervention_cost_total
        
        improvement_pct = ((optimized_net_revenue - baseline_total_revenue) / baseline_total_revenue) * 100
        
        print(f"\nOptimization Results (periods={periods}, budget={budget_constraint*100:.0f}%):")
        print(f"Baseline total revenue: ${baseline_total_revenue:,.2f}")
        print(f"Optimized gross revenue: ${optimized_total_revenue:,.2f}")
        print(f"Intervention cost: ${intervention_cost_total:,.2f}")
        print(f"Optimized net revenue: ${optimized_net_revenue:,.2f}")
        print(f"Net improvement: ${optimized_net_revenue - baseline_total_revenue:,.2f} ({improvement_pct:.2f}%)")
        print(f"Interventions scheduled: {intervention_count}/{periods} periods")
        
        # Format results
        baseline_results = {
            'state_history': baseline_state_history,
            'revenue_history': baseline_revenue_history,
            'total_revenue': baseline_total_revenue,
            'schedule': baseline_schedule
        }
        
        optimized_results = {
            'state_history': optimized_state_history,
            'revenue_history': optimized_revenue_history,
            'gross_revenue': optimized_total_revenue,
            'intervention_cost': intervention_cost_total,
            'net_revenue': optimized_net_revenue,
            'schedule': optimized_schedule,
            'improvement_pct': improvement_pct
        }
        
        # Visualize results
        self.visualize_optimization_results(
            baseline_results, 
            optimized_results, 
            periods, 
            population_size
        )
        
        return optimized_schedule, baseline_results, optimized_results
    
    def run_targeted_optimization(self, target_states=None, periods=12, population_size=10000):
        """
        Run separate optimizations targeting specific customer states
        
        Parameters:
        - target_states: List of states to target (if None, run for all states)
        - periods: Number of time periods to simulate
        - population_size: Number of customers to simulate
        
        Returns:
        - results: Dict with optimization results for each target state
        """
        if target_states is None:
            target_states = ['Premium', 'Value', 'Inactive']
        
        results = {}
        
        for state in target_states:
            print(f"\n{'='*80}")
            print(f"Optimizing for {state} customers")
            print(f"{'='*80}")
            
            # Create a custom effect matrix that only affects the target state
            original_effect_matrix = self.avg_effect_matrix.copy()
            
            # Create a targeted effect matrix
            targeted_effect_matrix = np.zeros_like(self.avg_effect_matrix)
            
            if state == 'Premium':
                # Only keep effects for Premium customers (row 0)
                targeted_effect_matrix[0, :] = original_effect_matrix[0, :]
            elif state == 'Value':
                # Only keep effects for Value customers (row 1)
                targeted_effect_matrix[1, :] = original_effect_matrix[1, :]
            elif state == 'Inactive':
                # Only keep effects for Inactive customers (row 2)
                targeted_effect_matrix[2, :] = original_effect_matrix[2, :]
            
            # Save original and set targeted matrix
            self._original_effect_matrix = self.avg_effect_matrix
            self.avg_effect_matrix = targeted_effect_matrix
            
            # Run optimization with targeted effect
            schedule, baseline, optimized = self.optimize_interventions(
                periods=periods,
                population_size=population_size,
                budget_constraint=0.3  # Allow up to 30% of periods to have interventions
            )
            
            # Store results
            results[state] = {
                'schedule': schedule,
                'baseline': baseline,
                'optimized': optimized
            }
            
            # Restore original effect matrix
            self.avg_effect_matrix = self._original_effect_matrix
        
        # Compare results across target states
        self.compare_targeted_strategies(results, periods, population_size)
        
        return results
    
    def visualize_optimization_results(self, baseline, optimized, periods, population_size):
        """Visualize optimization results"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})
        
        # Plot 1: Revenue comparison
        ax1 = axes[0]
        periods_range = np.arange(periods)
        
        # Convert to per-customer revenue for better scale
        baseline_revenue = baseline['revenue_history']
        optimized_revenue = optimized['revenue_history']
        
        ax1.plot(periods_range, baseline_revenue, 'b-', label='Baseline')
        ax1.plot(periods_range, optimized_revenue, 'g-', label='Optimized')
        
        # Mark intervention periods
        for t, intervention in optimized['schedule'].items():
            if intervention:
                ax1.axvline(x=t, color='r', linestyle='--', alpha=0.3)
        
        ax1.set_title('Revenue Comparison: Baseline vs. Optimized Interventions')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Revenue per Customer')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: State evolution comparison
        ax2 = axes[1]
        periods_range_ext = np.arange(periods + 1)
        
        # Set up bottom values for stacked area chart
        baseline_premium = baseline['state_history'][:, 0] * 100
        baseline_value = baseline['state_history'][:, 1] * 100
        baseline_inactive = baseline['state_history'][:, 2] * 100
        
        optimized_premium = optimized['state_history'][:, 0] * 100
        optimized_value = optimized['state_history'][:, 1] * 100
        optimized_inactive = optimized['state_history'][:, 2] * 100
        
        # Baseline stacked area (left side)
        ax2.stackplot(
            periods_range_ext, 
            [baseline_premium, baseline_value, baseline_inactive],
            labels=['Premium', 'Value', 'Inactive'],
            colors=['g', 'b', 'r'],
            alpha=0.4
        )
        
        # Add a vertical line to separate baseline and optimized
        ax2.axvline(x=periods, color='k', linestyle='-', alpha=0.7)
        
        # Optimized stacked area (right side)
        ax2.stackplot(
            periods_range_ext + periods + 1,  # Shift to right side
            [optimized_premium, optimized_value, optimized_inactive],
            colors=['g', 'b', 'r'],
            alpha=0.4
        )
        
        # Add text labels
        ax2.text(periods/2, 90, "BASELINE", ha='center', fontsize=12)
        ax2.text(periods + periods/2 + 1, 90, "OPTIMIZED", ha='center', fontsize=12)
        
        # Set x-axis labels
        ax2.set_xticks([])  # Remove x-ticks for cleaner look
        ax2.set_title('Customer State Evolution: Baseline vs. Optimized')
        ax2.set_ylabel('Percentage of Customers')
        ax2.set_ylim(0, 100)
        ax2.legend(loc='lower center')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'optimization_results.png'))
        plt.close()
        
        # Create intervention schedule visualization
        plt.figure(figsize=(12, 3))
        
        # Create binary intervention indicator
        interventions = np.zeros(periods)
        for t, val in optimized['schedule'].items():
            interventions[t] = val
        
        plt.bar(periods_range, interventions, color='orange', alpha=0.7)
        plt.title('Optimized Intervention Schedule')
        plt.xlabel('Time Period')
        plt.ylabel('Intervention (1=Yes, 0=No)')
        plt.yticks([0, 1])
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'intervention_schedule.png'))
        plt.close()
        
        # Save results to JSON (convert numpy arrays to lists for JSON serialization)
        results_json = {
            'baseline': {
                'revenue_history': baseline['revenue_history'].tolist(),
                'total_revenue': float(baseline['total_revenue']),
                'state_history': baseline['state_history'].tolist()
            },
            'optimized': {
                'revenue_history': optimized['revenue_history'].tolist(),
                'gross_revenue': float(optimized['gross_revenue']),
                'intervention_cost': float(optimized['intervention_cost']),
                'net_revenue': float(optimized['net_revenue']),
                'improvement_pct': float(optimized['improvement_pct']),
                'schedule': {str(k): v for k, v in optimized['schedule'].items()},
                'state_history': optimized['state_history'].tolist()
            },
            'parameters': {
                'periods': periods,
                'population_size': population_size,
                'revenue_by_state': self.revenue_by_state,
                'transition_matrix': self.transition_matrix.tolist(),
                'effect_matrix': self.avg_effect_matrix.tolist()
            }
        }
        
        with open(os.path.join(RESULTS_DIR, 'optimization_results.json'), 'w') as f:
            json.dump(results_json, f, indent=2)
    
    def compare_targeted_strategies(self, results, periods, population_size):
        """Compare optimization results across different target states"""
        # Extract key metrics for comparison
        metrics = {}
        for state, result in results.items():
            metrics[state] = {
                'net_revenue': result['optimized']['net_revenue'],
                'improvement_pct': result['optimized']['improvement_pct'],
                'intervention_count': sum(result['optimized']['schedule'].values())
            }
        
        # Create comparison visualizations
        plt.figure(figsize=(10, 6))
        
        # Bar chart of improvement percentages
        states = list(metrics.keys())
        improvements = [metrics[state]['improvement_pct'] for state in states]
        
        plt.bar(states, improvements, color='green', alpha=0.7)
        plt.title('Revenue Improvement by Target Strategy')
        plt.xlabel('Target Customer Segment')
        plt.ylabel('Improvement (%)')
        plt.grid(alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(improvements):
            plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'target_strategy_comparison.png'))
        plt.close()
        
        # Create schedule comparison
        plt.figure(figsize=(12, 6))
        
        # Plot intervention schedules for each target
        for i, (state, result) in enumerate(results.items()):
            schedule = result['optimized']['schedule']
            
            # Convert to array for plotting
            interventions = np.zeros(periods)
            for t, val in schedule.items():
                interventions[t] = val
            
            plt.plot(np.arange(periods), interventions + i*0.1, 'o-', label=f"Target: {state}")
        
        plt.title('Intervention Schedules by Target Strategy')
        plt.xlabel('Time Period')
        plt.ylabel('Intervention (offset for visibility)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.yticks([])  # Hide y-ticks for cleaner look
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'target_schedule_comparison.png'))
        plt.close()
        
        # Print comparison summary
        print("\nTarget Strategy Comparison:")
        print("-" * 60)
        print(f"{'Target':<10} {'Net Revenue':<15} {'Improvement':<15} {'Interventions':<15}")
        print("-" * 60)
        
        for state in states:
            print(f"{state:<10} ${metrics[state]['net_revenue']:,.2f}  {metrics[state]['improvement_pct']:>6.2f}%         {metrics[state]['intervention_count']}/{periods}")
        
        print("-" * 60)
        
        # Determine best strategy
        best_state = max(metrics, key=lambda x: metrics[x]['improvement_pct'])
        print(f"Best strategy: Target {best_state} customers ({metrics[best_state]['improvement_pct']:.2f}% improvement)")

def main():
    """Main function to run SIRS model optimization"""
    print("SIRS Model Intervention Optimization")
    print("=" * 40)
    
    # Create optimizer from the model
    optimizer = SIRSOptimizer()
    
    # Run base optimization
    print("\nRunning baseline optimization...")
    optimizer.optimize_interventions(periods=12, population_size=10000)
    
    # Run targeted optimization
    print("\nRunning targeted optimization...")
    optimizer.run_targeted_optimization(periods=12, population_size=10000)
    
    print("\nOptimization complete. Results saved to:", RESULTS_DIR)

if __name__ == "__main__":
    main() 