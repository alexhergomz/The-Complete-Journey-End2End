#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Customer Lifecycle Simulation Model
A statistical simulation model that replicates empirical patterns from segmentation
and enables optimization of key customer behavior variables
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Real

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'segmentation_results')
SIMULATION_DIR = os.path.join(BASE_DIR, 'simulation_results')
os.makedirs(SIMULATION_DIR, exist_ok=True)

class CustomerLifecycleModel:
    """Customer lifecycle simulation model"""
    
    def __init__(self, segmented_data_path=None):
        self.segments = {}
        self.transition_matrices = {}
        self.value_models = {}
        self.response_models = {}
        self.time_periods = 52  # Default to 52 weeks (1 year)
        
        # Load empirical data if path provided
        if segmented_data_path:
            self.load_empirical_data(segmented_data_path)
    
    def load_empirical_data(self, data_path):
        """Load segmented customer data from previous analysis"""
        print("Loading empirical customer data...")
        
        if os.path.exists(data_path):
            self.customer_data = pd.read_csv(data_path)
            print(f"Loaded data with {len(self.customer_data)} customers")
            
            # Extract segment information
            self.segment_ids = self.customer_data['cluster'].unique()
            
            # Extract key statistics by segment
            self.compute_segment_statistics()
        else:
            print(f"Error: Cannot find customer data at {data_path}")
    
    def compute_segment_statistics(self):
        """Compute key statistics for each segment from empirical data"""
        self.segment_stats = {}
        
        # Group data by cluster
        grouped = self.customer_data.groupby('cluster')
        
        for segment_id, segment_data in grouped:
            # Calculate key statistics
            stats = {
                'size': len(segment_data),
                'size_pct': len(segment_data) / len(self.customer_data),
                'avg_spend': segment_data['total_spend'].mean(),
                'avg_baskets': segment_data['total_baskets'].mean(),
                'avg_frequency': segment_data['basket_frequency'].mean(),
                'avg_products': segment_data['unique_products'].mean(),
                'avg_duration': segment_data['active_weeks'].mean()
            }
            
            # Calculate engagement probability (purchases per week)
            stats['engagement_prob'] = stats['avg_baskets'] / stats['avg_duration']
            
            # Calculate churn probability (simplified model based on frequency)
            stats['churn_baseline'] = 1 - (stats['avg_duration'] / 104)  # Assuming 2-year max span
            
            # Calculate response to marketing based on campaign participation
            if 'campaigns_participated' in segment_data.columns:
                stats['campaign_response'] = segment_data['coupons_redeemed'].sum() / segment_data['campaigns_participated'].sum()
            else:
                stats['campaign_response'] = 0.1  # Default fallback
                
            # Store segment statistics
            self.segment_stats[segment_id] = stats
            
            # Create probability distributions for key metrics
            self._fit_distributions(segment_id, segment_data)
    
    def _fit_distributions(self, segment_id, segment_data):
        """Fit probability distributions to empirical data for simulation"""
        distributions = {}
        
        # Basket value distribution
        if 'avg_transaction_value' in segment_data.columns:
            # Remove outliers for better distribution fit
            avg_basket_values = segment_data['avg_transaction_value']
            avg_basket_values = avg_basket_values[avg_basket_values < avg_basket_values.quantile(0.99)]
            
            # Fit gamma distribution (common for monetary values)
            params = stats.gamma.fit(avg_basket_values.dropna())
            distributions['basket_value'] = {
                'dist': stats.gamma,
                'params': params
            }
        
        # Inter-purchase time distribution
        if 'basket_frequency' in segment_data.columns:
            # Convert frequency to time between purchases
            frequency = segment_data['basket_frequency']
            inter_purchase_time = 1 / frequency[frequency > 0]
            
            # Fit exponential distribution (common for time between events)
            params = stats.expon.fit(inter_purchase_time)
            distributions['inter_purchase_time'] = {
                'dist': stats.expon,
                'params': params
            }
        
        # Product diversity distribution
        if 'unique_products' in segment_data.columns:
            product_counts = segment_data['unique_products']
            
            # Fit normal distribution
            params = stats.norm.fit(product_counts)
            distributions['product_count'] = {
                'dist': stats.norm,
                'params': params
            }
        
        # Store distributions for this segment
        self.segments[segment_id] = distributions
    
    def build_transition_matrices(self):
        """Build customer state transition matrices for each segment"""
        customer_states = ['active', 'at_risk', 'dormant', 'churned']
        
        for segment_id, stats in self.segment_stats.items():
            # Base transition probabilities derived from segment statistics
            # Format: transition[from_state][to_state]
            transitions = np.zeros((4, 4))
            
            # From active
            transitions[0, 0] = 0.80  # Stay active
            transitions[0, 1] = 0.15  # Become at-risk
            transitions[0, 2] = 0.04  # Become dormant
            transitions[0, 3] = 0.01  # Churn
            
            # From at-risk
            transitions[1, 0] = 0.30  # Recover to active
            transitions[1, 1] = 0.40  # Stay at-risk
            transitions[1, 2] = 0.25  # Become dormant
            transitions[1, 3] = 0.05  # Churn
            
            # From dormant
            transitions[2, 0] = 0.10  # Recover to active
            transitions[2, 1] = 0.20  # Become at-risk
            transitions[2, 2] = 0.50  # Stay dormant
            transitions[2, 3] = 0.20  # Churn
            
            # From churned - assume a small win-back probability
            transitions[3, 0] = 0.01  # Win back to active
            transitions[3, 1] = 0.02  # Win back to at-risk
            transitions[3, 2] = 0.07  # Win back to dormant
            transitions[3, 3] = 0.90  # Stay churned
            
            # Adjust based on segment characteristics
            # Value shoppers (0) have higher churn risk
            if segment_id == 0:
                transitions[0, 3] *= 1.5
                transitions[1, 3] *= 1.5
                transitions[0, 0] -= transitions[0, 3] - 0.01
                transitions[1, 1] -= transitions[1, 3] - 0.05
            
            # Premium shoppers (1) have higher loyalty
            elif segment_id == 1:
                transitions[0, 0] += 0.05
                transitions[1, 0] += 0.10
                transitions[0, 3] *= 0.5
                transitions[1, 3] *= 0.7
                
                # Normalize rows to sum to 1.0
                for i in range(4):
                    transitions[i] = transitions[i] / transitions[i].sum()
            
            self.transition_matrices[segment_id] = {
                'states': customer_states,
                'matrix': transitions
            }
    
    def build_response_models(self):
        """Build models for customer response to marketing interventions"""
        for segment_id, stats in self.segment_stats.items():
            # Base response rates by customer state
            base_responses = {
                'active': {
                    'frequency_campaign': 0.15,  # Probability of increasing visit frequency
                    'basket_campaign': 0.20,     # Probability of increasing basket value
                    'cross_sell_campaign': 0.25  # Probability of buying new category
                },
                'at_risk': {
                    'frequency_campaign': 0.10,
                    'basket_campaign': 0.15,
                    'cross_sell_campaign': 0.10
                },
                'dormant': {
                    'frequency_campaign': 0.05,
                    'basket_campaign': 0.05,
                    'cross_sell_campaign': 0.03
                },
                'churned': {
                    'frequency_campaign': 0.01,
                    'basket_campaign': 0.01,
                    'cross_sell_campaign': 0.01
                }
            }
            
            # Adjust based on segment characteristics
            response_multiplier = stats['campaign_response'] / 0.15  # Normalize to a reasonable baseline
            
            # Apply segment-specific adjustments
            for state in base_responses:
                for campaign in base_responses[state]:
                    base_responses[state][campaign] *= response_multiplier
                    
                    # Cap probabilities at reasonable values
                    base_responses[state][campaign] = min(base_responses[state][campaign], 0.9)
            
            self.response_models[segment_id] = base_responses
    
    def build_value_models(self):
        """Build models for customer value evolution"""
        for segment_id, stats in self.segment_stats.items():
            # Value model based on:
            # - Base spending rate (per time period)
            # - State-dependent multipliers
            # - Response effects
            
            # Base spending derived from empirical data
            base_period_spend = stats['avg_spend'] / stats['avg_duration']
            
            # Value multipliers by state
            value_multipliers = {
                'active': 1.0,
                'at_risk': 0.7,
                'dormant': 0.2,
                'churned': 0.0
            }
            
            # Effect sizes for different intervention types
            intervention_effects = {
                'frequency_campaign': {
                    'success': 1.2,  # 20% lift in spend if successful
                    'failure': 1.0   # No change if unsuccessful
                },
                'basket_campaign': {
                    'success': 1.3,  # 30% lift in spend if successful
                    'failure': 1.0
                },
                'cross_sell_campaign': {
                    'success': 1.25,  # 25% lift in spend if successful
                    'failure': 1.0
                }
            }
            
            self.value_models[segment_id] = {
                'base_spend': base_period_spend,
                'multipliers': value_multipliers,
                'intervention_effects': intervention_effects
            }
    
    def initialize_simulation(self, population_size=10000, time_periods=52):
        """Initialize the customer population for simulation"""
        self.time_periods = time_periods
        self.population = []
        
        # Create initial population based on segment sizes
        for segment_id, stats in self.segment_stats.items():
            segment_size = int(stats['size_pct'] * population_size)
            
            for i in range(segment_size):
                # Generate initial customer state (most start as active)
                initial_state = np.random.choice(
                    ['active', 'at_risk', 'dormant', 'churned'],
                    p=[0.7, 0.2, 0.08, 0.02]
                )
                
                # Sample initial spending based on segment distributions
                if segment_id in self.segments and 'basket_value' in self.segments[segment_id]:
                    dist = self.segments[segment_id]['basket_value']
                    initial_spend = dist['dist'].rvs(*dist['params'][:-2], 
                                                    loc=dist['params'][-2], 
                                                    scale=dist['params'][-1])
                else:
                    # Fallback to average if distribution not available
                    initial_spend = stats['avg_spend'] / stats['avg_duration']
                
                # Create customer
                customer = {
                    'id': len(self.population),
                    'segment': segment_id,
                    'state': initial_state,
                    'spend_rate': max(0, initial_spend),  # Ensure non-negative
                    'total_spend': 0,
                    'weeks_active': 0,
                    'purchases': 0,
                    'state_history': [],
                    'spend_history': [],
                    'response_history': []
                }
                
                self.population.append(customer)
        
        print(f"Initialized population of {len(self.population)} customers across {len(self.segment_stats)} segments")
    
    def run_simulation(self, intervention_schedule=None):
        """Run customer lifecycle simulation"""
        print(f"Running simulation for {self.time_periods} periods...")
        
        # Initialize results storage
        self.results = {
            'total_spend': np.zeros(self.time_periods),
            'active_customers': np.zeros(self.time_periods),
            'at_risk_customers': np.zeros(self.time_periods),
            'dormant_customers': np.zeros(self.time_periods),
            'churned_customers': np.zeros(self.time_periods),
            'segment_spend': {segment_id: np.zeros(self.time_periods) 
                             for segment_id in self.segment_stats.keys()}
        }
        
        # Default empty intervention schedule if none provided
        if intervention_schedule is None:
            intervention_schedule = {}
        
        # Run simulation for each time period
        for t in tqdm(range(self.time_periods)):
            # Apply scheduled interventions for this period
            current_interventions = intervention_schedule.get(t, [])
            
            # Process each customer
            for customer in self.population:
                segment_id = customer['segment']
                current_state = customer['state']
                
                # Record state
                customer['state_history'].append(current_state)
                
                # Only active customers generate spending
                period_spend = 0
                if current_state == 'active':
                    # Base spending from value model
                    base_spend = self.value_models[segment_id]['base_spend']
                    state_multiplier = self.value_models[segment_id]['multipliers'][current_state]
                    
                    # Apply any relevant intervention effects
                    intervention_multiplier = 1.0
                    for intervention in current_interventions:
                        # Check if customer responds to intervention
                        intervention_type = intervention['type']
                        response_prob = self.response_models[segment_id][current_state][intervention_type]
                        
                        # Determine if intervention was successful
                        success = np.random.random() < response_prob
                        
                        # Record response
                        customer['response_history'].append({
                            'period': t,
                            'intervention': intervention_type,
                            'success': success
                        })
                        
                        # Apply effect if successful
                        effect_key = 'success' if success else 'failure'
                        intervention_multiplier *= self.value_models[segment_id]['intervention_effects'][intervention_type][effect_key]
                    
                    # Calculate final spend
                    period_spend = base_spend * state_multiplier * intervention_multiplier
                    
                    # Add random variation (lognormal for skewed spending)
                    noise = np.random.lognormal(0, 0.3)  # Mean 1.0 with some variance
                    period_spend *= noise
                    
                    # Update customer metrics
                    customer['purchases'] += 1
                    customer['weeks_active'] += 1
                
                # Record spending
                customer['spend_history'].append(period_spend)
                customer['total_spend'] += period_spend
                
                # Update aggregate results
                self.results['total_spend'][t] += period_spend
                self.results['segment_spend'][segment_id][t] += period_spend
                
                # Transition to next state
                next_state_idx = np.random.choice(
                    4,  # Four states: active, at_risk, dormant, churned
                    p=self.transition_matrices[segment_id]['matrix'][
                        self.transition_matrices[segment_id]['states'].index(current_state)
                    ]
                )
                customer['state'] = self.transition_matrices[segment_id]['states'][next_state_idx]
            
            # Count customers in each state after transitions
            state_counts = {'active': 0, 'at_risk': 0, 'dormant': 0, 'churned': 0}
            for customer in self.population:
                state_counts[customer['state']] += 1
            
            # Update state counts in results
            self.results['active_customers'][t] = state_counts['active']
            self.results['at_risk_customers'][t] = state_counts['at_risk']
            self.results['dormant_customers'][t] = state_counts['dormant']
            self.results['churned_customers'][t] = state_counts['churned']
        
        print("Simulation completed.")
        return self.results
    
    def visualize_results(self):
        """Visualize simulation results"""
        if not hasattr(self, 'results'):
            print("No simulation results to visualize. Run simulation first.")
            return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Revenue over time
        ax1 = axes[0, 0]
        ax1.plot(self.results['total_spend'], linewidth=2)
        ax1.set_title('Total Revenue Over Time')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Revenue')
        ax1.grid(alpha=0.3)
        
        # Plot 2: Customer state evolution
        ax2 = axes[0, 1]
        ax2.plot(self.results['active_customers'], 'g-', label='Active')
        ax2.plot(self.results['at_risk_customers'], 'y-', label='At Risk')
        ax2.plot(self.results['dormant_customers'], 'r-', label='Dormant')
        ax2.plot(self.results['churned_customers'], 'k-', label='Churned')
        ax2.set_title('Customer States Over Time')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Number of Customers')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: Segment revenues
        ax3 = axes[1, 0]
        for segment_id, spend in self.results['segment_spend'].items():
            label = f"Segment {segment_id}" 
            if segment_id == 0:
                label = "Value Shoppers"
            elif segment_id == 1:
                label = "Premium Shoppers"
            ax3.plot(spend, label=label)
        ax3.set_title('Segment Revenues Over Time')
        ax3.set_xlabel('Time Period')
        ax3.set_ylabel('Revenue')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Plot 4: Customer state proportions
        ax4 = axes[1, 1]
        total_customers = len(self.population)
        active_pct = self.results['active_customers'] / total_customers * 100
        at_risk_pct = self.results['at_risk_customers'] / total_customers * 100
        dormant_pct = self.results['dormant_customers'] / total_customers * 100
        churned_pct = self.results['churned_customers'] / total_customers * 100
        
        ax4.stackplot(range(self.time_periods),
                     active_pct, at_risk_pct, dormant_pct, churned_pct,
                     labels=['Active', 'At Risk', 'Dormant', 'Churned'],
                     colors=['g', 'y', 'r', 'k'], alpha=0.7)
        ax4.set_title('Customer State Proportions')
        ax4.set_xlabel('Time Period')
        ax4.set_ylabel('Percentage of Customers')
        ax4.legend(loc='upper left')
        ax4.set_ylim(0, 100)
        ax4.grid(alpha=0.3)
        
        # Save and show figure
        plt.tight_layout()
        plt.savefig(os.path.join(SIMULATION_DIR, 'simulation_results.png'))
        plt.close()
        
        print("Visualization saved to simulation_results directory.")
    
    def optimize_interventions(self, budget_constraint=1.0, optimization_goal='revenue'):
        """Optimize the intervention schedule to maximize a goal within budget constraint"""
        print("Optimizing intervention schedule...")
        
        # Define intervention types and their costs (relative units)
        intervention_types = {
            'frequency_campaign': {'cost': 0.2, 'max_per_period': 1},
            'basket_campaign': {'cost': 0.3, 'max_per_period': 1},
            'cross_sell_campaign': {'cost': 0.4, 'max_per_period': 1}
        }
        
        # Number of time periods to optimize (we'll use a shorter horizon for optimization)
        optimization_periods = min(26, self.time_periods)
        
        # Define objective function for optimization (negative because we're minimizing)
        def objective_function(x):
            # Convert flat array to intervention schedule
            intervention_schedule = {}
            idx = 0
            for t in range(optimization_periods):
                interventions = []
                for intervention_type, specs in intervention_types.items():
                    # If the binary flag is 1, add this intervention
                    if x[idx] > 0.5:  # Threshold for binary approximation
                        interventions.append({'type': intervention_type, 'period': t})
                    idx += 1
                
                if interventions:
                    intervention_schedule[t] = interventions
            
            # Run simulation with this schedule
            self.initialize_simulation(population_size=1000, time_periods=optimization_periods)
            results = self.run_simulation(intervention_schedule)
            
            # Calculate total cost
            total_cost = 0
            for t, interventions in intervention_schedule.items():
                for intervention in interventions:
                    total_cost += intervention_types[intervention['type']]['cost']
            
            # Define goal metric
            if optimization_goal == 'revenue':
                goal_value = results['total_spend'].sum()
            elif optimization_goal == 'retention':
                goal_value = -results['churned_customers'][-1]  # Negative because we want to minimize churn
            else:
                goal_value = results['total_spend'].sum()  # Default to revenue
            
            # Penalize if budget constraint is violated
            if total_cost > budget_constraint:
                penalty = 1000 * (total_cost - budget_constraint)
                goal_value -= penalty
            
            # Return negative value (since we're minimizing)
            return -goal_value
        
        # Initial solution (no interventions)
        initial_solution = np.zeros(len(intervention_types) * optimization_periods)
        
        # Constraints: binary values
        bounds = [(0, 1) for _ in range(len(initial_solution))]
        
        # Run optimization
        result = minimize(
            objective_function,
            initial_solution,
            method='SLSQP',
            bounds=bounds,
            options={'disp': True, 'maxiter': 50}
        )
        
        # Convert result to intervention schedule
        optimized_schedule = {}
        idx = 0
        for t in range(optimization_periods):
            interventions = []
            for intervention_type, specs in intervention_types.items():
                if result.x[idx] > 0.5:
                    interventions.append({'type': intervention_type, 'period': t})
                idx += 1
            
            if interventions:
                optimized_schedule[t] = interventions
        
        # Calculate schedule cost
        schedule_cost = 0
        for t, interventions in optimized_schedule.items():
            for intervention in interventions:
                schedule_cost += intervention_types[intervention['type']]['cost']
        
        print(f"Optimization complete. Schedule cost: {schedule_cost:.2f} units (budget: {budget_constraint:.2f})")
        
        # Run full simulation with optimized schedule
        self.initialize_simulation(time_periods=self.time_periods)
        optimized_results = self.run_simulation(optimized_schedule)
        
        # Run baseline simulation for comparison
        self.initialize_simulation(time_periods=self.time_periods)
        baseline_results = self.run_simulation()
        
        # Calculate improvement
        baseline_metric = baseline_results['total_spend'].sum()
        optimized_metric = optimized_results['total_spend'].sum()
        improvement = (optimized_metric - baseline_metric) / baseline_metric * 100
        
        print(f"Baseline total revenue: ${baseline_metric:.2f}")
        print(f"Optimized total revenue: ${optimized_metric:.2f}")
        print(f"Improvement: {improvement:.2f}%")
        
        # Visualize comparison
        plt.figure(figsize=(12, 8))
        plt.plot(baseline_results['total_spend'], 'b-', label='Baseline')
        plt.plot(optimized_results['total_spend'], 'g-', label='Optimized')
        plt.title('Revenue Comparison: Baseline vs. Optimized Interventions')
        plt.xlabel('Time Period')
        plt.ylabel('Revenue')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Mark intervention periods on the chart
        for t in optimized_schedule.keys():
            if t < self.time_periods:
                intervention_types_in_period = [i['type'] for i in optimized_schedule[t]]
                legend_text = ", ".join([i.split('_')[0] for i in intervention_types_in_period])
                plt.axvline(x=t, color='r', linestyle='--', alpha=0.3)
                plt.text(t, optimized_results['total_spend'][t], legend_text, 
                         fontsize=8, rotation=90, verticalalignment='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(SIMULATION_DIR, 'optimization_comparison.png'))
        plt.close()
        
        # Save optimization details
        with open(os.path.join(SIMULATION_DIR, 'optimization_details.txt'), 'w') as f:
            f.write("Optimization Results\n")
            f.write("====================\n\n")
            f.write(f"Optimization goal: {optimization_goal}\n")
            f.write(f"Budget constraint: {budget_constraint} units\n")
            f.write(f"Schedule cost: {schedule_cost:.2f} units\n\n")
            
            f.write("Intervention Schedule:\n")
            for t in sorted(optimized_schedule.keys()):
                f.write(f"Period {t}: {[i['type'] for i in optimized_schedule[t]]}\n")
            
            f.write("\nResults:\n")
            f.write(f"Baseline total revenue: ${baseline_metric:.2f}\n")
            f.write(f"Optimized total revenue: ${optimized_metric:.2f}\n")
            f.write(f"Improvement: {improvement:.2f}%\n")
        
        return optimized_schedule, optimized_results, baseline_results
    
    def optimize_segment_variables(self, segment_id, n_simulations=100, population_size=5000):
        """
        Optimize significant variables for a specific customer segment using machine learning
        
        Parameters:
        - segment_id: ID of the segment to optimize (0 for Value Shoppers, 1 for Premium Shoppers)
        - n_simulations: Number of simulations to run for training the ML model
        - population_size: Size of the simulated population for each run
        
        Returns:
        - Dictionary of optimized variable values
        """
        print(f"Starting optimization for segment {segment_id}")
        
        # Define the variables to optimize and their ranges based on segment
        if segment_id == 0:  # Value Shoppers
            # Based on regression coefficients and correlation analysis
            param_space = [
                Real(0.5, 2.0, name='top_dept_spend_multiplier'),    # Multiplier for top department spend
                Real(0.5, 2.0, name='total_items_multiplier'),       # Multiplier for total items
                Real(0.5, 2.0, name='unique_products_multiplier'),   # Multiplier for unique products
                Real(0.5, 2.0, name='basket_frequency_multiplier'),  # Multiplier for basket frequency
                Real(0.5, 2.0, name='avg_basket_value_multiplier')   # Multiplier for avg basket value
            ]
        else:  # Premium Shoppers (segment_id == 1)
            param_space = [
                Real(0.5, 2.0, name='top_dept_spend_multiplier'),        # Multiplier for top department spend
                Real(0.5, 2.0, name='unique_products_multiplier'),       # Multiplier for unique products
                Real(0.5, 2.0, name='campaigns_participated_multiplier'),# Multiplier for campaign participation
                Real(0.5, 2.0, name='active_weeks_multiplier'),          # Multiplier for active weeks
                Real(0.5, 2.0, name='total_baskets_multiplier')          # Multiplier for total baskets
            ]
        
        # Generate training data by running simulations with different parameter values
        X_train = []
        y_train = []
        
        print("Generating simulation data for ML model training...")
        for _ in range(n_simulations):
            # Randomly sample parameters within ranges
            if segment_id == 0:  # Value Shoppers
                params = [
                    np.random.uniform(0.5, 2.0),  # top_dept_spend_multiplier
                    np.random.uniform(0.5, 2.0),  # total_items_multiplier
                    np.random.uniform(0.5, 2.0),  # unique_products_multiplier
                    np.random.uniform(0.5, 2.0),  # basket_frequency_multiplier
                    np.random.uniform(0.5, 2.0)   # avg_basket_value_multiplier
                ]
            else:  # Premium Shoppers
                params = [
                    np.random.uniform(0.5, 2.0),  # top_dept_spend_multiplier
                    np.random.uniform(0.5, 2.0),  # unique_products_multiplier
                    np.random.uniform(0.5, 2.0),  # campaigns_participated_multiplier
                    np.random.uniform(0.5, 2.0),  # active_weeks_multiplier
                    np.random.uniform(0.5, 2.0)   # total_baskets_multiplier
                ]
            
            # Apply these parameters to the model
            self._apply_optimization_params(segment_id, params)
            
            # Run simulation with this parameter set
            self.initialize_simulation(population_size=population_size, time_periods=26)
            results = self.run_simulation()
            
            # Extract average revenue for the specific segment
            segment_revenue = np.mean(results['segment_spend'][segment_id])
            
            # Store for training
            X_train.append(params)
            y_train.append(segment_revenue)
        
        # Train a GradientBoostingRegressor model
        print("Training GradientBoostingRegressor model on simulation data...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_scaled, y_train)
        
        # Display feature importances
        print("\nFeature Importances:")
        if segment_id == 0:  # Value Shoppers
            features = ['Top Dept Spend', 'Total Items', 'Unique Products', 'Basket Frequency', 'Avg Basket Value']
        else:  # Premium Shoppers
            features = ['Top Dept Spend', 'Unique Products', 'Campaign Participation', 'Active Weeks', 'Total Baskets']
        
        for i, importance in enumerate(model.feature_importances_):
            print(f"{features[i]}: {importance:.4f}")
        
        # Use Bayesian optimization to find optimal parameters
        print("\nRunning Bayesian optimization to find optimal parameter values...")
        
        def objective_function(params):
            # Scale the params
            params_scaled = scaler.transform([params])[0]
            # Predict revenue using the trained model
            predicted_revenue = model.predict([params_scaled])[0]
            # Return negative revenue for minimization
            return -predicted_revenue
        
        # Run optimization
        result = gp_minimize(
            objective_function,
            param_space,
            n_calls=50,
            random_state=42,
            verbose=True
        )
        
        # Get optimized parameters
        optimized_params = result.x
        
        # Format results
        optimized_values = {}
        if segment_id == 0:  # Value Shoppers
            optimized_values = {
                'top_dept_spend_multiplier': optimized_params[0],
                'total_items_multiplier': optimized_params[1],
                'unique_products_multiplier': optimized_params[2],
                'basket_frequency_multiplier': optimized_params[3],
                'avg_basket_value_multiplier': optimized_params[4]
            }
        else:  # Premium Shoppers
            optimized_values = {
                'top_dept_spend_multiplier': optimized_params[0],
                'unique_products_multiplier': optimized_params[1],
                'campaigns_participated_multiplier': optimized_params[2],
                'active_weeks_multiplier': optimized_params[3],
                'total_baskets_multiplier': optimized_params[4]
            }
        
        # Verify with a final simulation using optimized parameters
        print("\nVerifying optimized parameters with a final simulation...")
        self._apply_optimization_params(segment_id, optimized_params)
        
        self.initialize_simulation(population_size=population_size, time_periods=52)
        optimized_results = self.run_simulation()
        
        # Reset the model parameters to their original values
        self._reset_optimization_params(segment_id)
        
        # Run a baseline simulation for comparison
        self.initialize_simulation(population_size=population_size, time_periods=52)
        baseline_results = self.run_simulation()
        
        # Calculate improvement
        baseline_revenue = np.mean(baseline_results['segment_spend'][segment_id])
        optimized_revenue = np.mean(optimized_results['segment_spend'][segment_id])
        improvement = (optimized_revenue - baseline_revenue) / baseline_revenue * 100
        
        print(f"\nBaseline average revenue for segment {segment_id}: ${baseline_revenue:.2f}")
        print(f"Optimized average revenue for segment {segment_id}: ${optimized_revenue:.2f}")
        print(f"Improvement: {improvement:.2f}%")
        
        # Create a visualization comparing baseline and optimized
        plt.figure(figsize=(12, 6))
        plt.plot(baseline_results['segment_spend'][segment_id], 'b-', label='Baseline')
        plt.plot(optimized_results['segment_spend'][segment_id], 'g-', label='Optimized')
        plt.title(f'Segment {segment_id} Revenue: Baseline vs. Optimized Variables')
        plt.xlabel('Time Period')
        plt.ylabel('Revenue')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(SIMULATION_DIR, f'segment_{segment_id}_optimization.png'))
        plt.close()
        
        return optimized_values
    
    def _apply_optimization_params(self, segment_id, params):
        """Apply optimization parameters to the model for a specific segment"""
        # Store original values to reset later
        if not hasattr(self, '_original_values'):
            self._original_values = {}
        
        self._original_values[segment_id] = {
            'value_model': self.value_models.get(segment_id, {}).copy(),
            'response_model': self.response_models.get(segment_id, {}).copy(),
            'transition_matrix': self.transition_matrices.get(segment_id, {}).copy()
        }
        
        # Apply parameter changes based on segment
        if segment_id == 0:  # Value Shoppers
            # 1. Top department spend -> affects base_spend
            if segment_id in self.value_models:
                self.value_models[segment_id]['base_spend'] *= params[0]
            
            # 2. Total items -> affects intervention effects
            if segment_id in self.value_models:
                for campaign in self.value_models[segment_id]['intervention_effects']:
                    self.value_models[segment_id]['intervention_effects'][campaign]['success'] *= params[1]
            
            # 3. Unique products -> affects response rates
            if segment_id in self.response_models:
                for state in self.response_models[segment_id]:
                    for campaign in self.response_models[segment_id][state]:
                        self.response_models[segment_id][state][campaign] *= params[2]
            
            # 4. Basket frequency -> affects transition probabilities (active state retention)
            if segment_id in self.transition_matrices:
                matrix = self.transition_matrices[segment_id]['matrix']
                # Increase probability of staying active
                matrix[0, 0] = min(0.95, matrix[0, 0] * params[3])
                # Normalize row
                matrix[0] = matrix[0] / matrix[0].sum()
            
            # 5. Average basket value -> affects value multipliers
            if segment_id in self.value_models:
                for state in self.value_models[segment_id]['multipliers']:
                    self.value_models[segment_id]['multipliers'][state] *= params[4]
        
        else:  # Premium Shoppers (segment_id == 1)
            # 1. Top department spend -> affects base_spend
            if segment_id in self.value_models:
                self.value_models[segment_id]['base_spend'] *= params[0]
            
            # 2. Unique products -> affects response rates
            if segment_id in self.response_models:
                for state in self.response_models[segment_id]:
                    for campaign in self.response_models[segment_id][state]:
                        self.response_models[segment_id][state][campaign] *= params[1]
            
            # 3. Campaign participation -> affects intervention effects
            if segment_id in self.value_models:
                for campaign in self.value_models[segment_id]['intervention_effects']:
                    self.value_models[segment_id]['intervention_effects'][campaign]['success'] *= params[2]
            
            # 4. Active weeks -> affects transition probabilities (retention)
            if segment_id in self.transition_matrices:
                matrix = self.transition_matrices[segment_id]['matrix']
                # Increase probability of staying active
                matrix[0, 0] = min(0.95, matrix[0, 0] * params[3])
                # Decrease probability of churning
                matrix[0, 3] = max(0.005, matrix[0, 3] / params[3])
                # Normalize row
                matrix[0] = matrix[0] / matrix[0].sum()
            
            # 5. Total baskets -> affects value multipliers
            if segment_id in self.value_models:
                for state in self.value_models[segment_id]['multipliers']:
                    self.value_models[segment_id]['multipliers'][state] *= params[4]
    
    def _reset_optimization_params(self, segment_id):
        """Reset the model parameters to their original values for a segment"""
        if hasattr(self, '_original_values') and segment_id in self._original_values:
            if 'value_model' in self._original_values[segment_id]:
                self.value_models[segment_id] = self._original_values[segment_id]['value_model'].copy()
            
            if 'response_model' in self._original_values[segment_id]:
                self.response_models[segment_id] = self._original_values[segment_id]['response_model'].copy()
            
            if 'transition_matrix' in self._original_values[segment_id]:
                self.transition_matrices[segment_id] = self._original_values[segment_id]['transition_matrix'].copy()

def main():
    """Main function to run the customer lifecycle simulation"""
    # Initialize model from empirical data
    model = CustomerLifecycleModel(os.path.join(DATA_DIR, 'customer_segments.csv'))
    
    # Build component models
    model.build_transition_matrices()
    model.build_response_models()
    model.build_value_models()
    
    # Run baseline simulation
    model.initialize_simulation(time_periods=52)  # 1 year simulation
    baseline_results = model.run_simulation()
    model.visualize_results()
    
    # Run optimization
    optimized_schedule, optimized_results, _ = model.optimize_interventions(
        budget_constraint=5.0,
        optimization_goal='revenue'
    )
    
    # Print the optimized schedule
    print("\nOptimized Intervention Schedule:")
    for period, interventions in sorted(optimized_schedule.items()):
        intervention_names = [i['type'] for i in interventions]
        print(f"Period {period}: {intervention_names}")
    
    print("\nSimulation and optimization complete. Results saved to the 'simulation_results' directory.")

if __name__ == "__main__":
    main() 