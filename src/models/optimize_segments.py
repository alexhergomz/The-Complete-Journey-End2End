#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Segment-Specific Variable Optimization for Customer Lifecycle Model
Optimizes the key variables for each customer segment using ML-based approach
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from customer_lifecycle_simulation import CustomerLifecycleModel

def main():
    """Main function to run segment-specific optimization"""
    
    # Create output directory for results
    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'optimization_results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            'segmentation_results', 'customer_segments.csv')
    model = CustomerLifecycleModel(model_path)
    
    # Build component models
    model.build_transition_matrices()
    model.build_response_models()
    model.build_value_models()
    
    # Optimize for Value Shoppers (Segment 0)
    print("\n" + "="*80)
    print("OPTIMIZING FOR VALUE SHOPPERS (SEGMENT 0)")
    print("="*80)
    value_optimized = model.optimize_segment_variables(
        segment_id=0,
        n_simulations=50,  # Reduced for faster execution, increase for better results
        population_size=3000
    )
    
    # Optimize for Premium Shoppers (Segment 1)
    print("\n" + "="*80)
    print("OPTIMIZING FOR PREMIUM SHOPPERS (SEGMENT 1)")
    print("="*80)
    premium_optimized = model.optimize_segment_variables(
        segment_id=1,
        n_simulations=50,  # Reduced for faster execution, increase for better results
        population_size=3000
    )
    
    # Save optimization results
    results = {
        'Value Shoppers (Segment 0)': value_optimized,
        'Premium Shoppers (Segment 1)': premium_optimized
    }
    
    # Create a DataFrame for better visualization
    rows = []
    for segment, params in results.items():
        for param, value in params.items():
            rows.append({
                'Segment': segment,
                'Parameter': param,
                'Optimized Value': value,
                'Target Action': _get_action_recommendation(param, value)
            })
    
    results_df = pd.DataFrame(rows)
    
    # Save results to CSV
    results_path = os.path.join(RESULTS_DIR, 'segment_optimization_results.csv')
    results_df.to_csv(results_path, index=False)
    
    # Display results
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print(f"Results saved to: {results_path}")
    print("\nValue Shoppers (Segment 0) Optimal Values:")
    for param, value in value_optimized.items():
        action = _get_action_recommendation(param, value)
        print(f"  {param}: {value:.4f} → {action}")
    
    print("\nPremium Shoppers (Segment 1) Optimal Values:")
    for param, value in premium_optimized.items():
        action = _get_action_recommendation(param, value)
        print(f"  {param}: {value:.4f} → {action}")
    
    # Create comparison visualization
    _create_optimization_visualization(results, RESULTS_DIR)

def _get_action_recommendation(param, value):
    """Convert optimization parameter values to actionable recommendations"""
    
    # Base change percentage
    change_pct = (value - 1.0) * 100
    direction = "increase" if change_pct > 0 else "decrease"
    abs_change = abs(change_pct)
    
    # Parameter specific recommendations
    if 'top_dept_spend' in param:
        if abs_change < 5:
            return "Maintain current category focus"
        elif abs_change > 20:
            return f"Strongly {direction} focus on top department by {abs_change:.1f}%"
        else:
            return f"{direction.capitalize()} focus on top department by {abs_change:.1f}%"
    
    elif 'total_items' in param:
        if abs_change < 5:
            return "Maintain current basket size"
        elif abs_change > 20:
            return f"Strongly {direction} items per transaction by {abs_change:.1f}%"
        else:
            return f"{direction.capitalize()} items per transaction by {abs_change:.1f}%"
    
    elif 'unique_products' in param:
        if abs_change < 5:
            return "Maintain current product diversity"
        elif abs_change > 20:
            return f"Strongly {direction} product variety by {abs_change:.1f}%"
        else:
            return f"{direction.capitalize()} product variety by {abs_change:.1f}%"
    
    elif 'basket_frequency' in param:
        if abs_change < 5:
            return "Maintain current visit frequency"
        elif abs_change > 20:
            return f"Strongly {direction} visit frequency by {abs_change:.1f}%"
        else:
            return f"{direction.capitalize()} visit frequency by {abs_change:.1f}%"
    
    elif 'avg_basket_value' in param:
        if abs_change < 5:
            return "Maintain current spending level"
        elif abs_change > 20:
            return f"Strongly {direction} transaction value by {abs_change:.1f}%"
        else:
            return f"{direction.capitalize()} transaction value by {abs_change:.1f}%"
    
    elif 'campaigns_participated' in param:
        if abs_change < 5:
            return "Maintain current campaign participation"
        elif abs_change > 20:
            return f"Strongly {direction} targeted campaigns by {abs_change:.1f}%"
        else:
            return f"{direction.capitalize()} targeted campaigns by {abs_change:.1f}%"
    
    elif 'active_weeks' in param:
        if abs_change < 5:
            return "Maintain current customer activity duration"
        elif abs_change > 20:
            return f"Strongly {direction} active shopping period by {abs_change:.1f}%"
        else:
            return f"{direction.capitalize()} active shopping period by {abs_change:.1f}%"
    
    elif 'total_baskets' in param:
        if abs_change < 5:
            return "Maintain current number of transactions"
        elif abs_change > 20:
            return f"Strongly {direction} number of transactions by {abs_change:.1f}%"
        else:
            return f"{direction.capitalize()} number of transactions by {abs_change:.1f}%"
    
    else:
        if abs_change < 5:
            return "Minimal change needed"
        else:
            return f"{direction.capitalize()} by {abs_change:.1f}%"

def _create_optimization_visualization(results, output_dir):
    """Create visualizations comparing optimization results"""
    
    # Create a bar chart comparing parameter values between segments
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Organize data for plotting
    value_params = list(results['Value Shoppers (Segment 0)'].keys())
    value_values = list(results['Value Shoppers (Segment 0)'].values())
    
    premium_params = list(results['Premium Shoppers (Segment 1)'].keys())
    premium_values = list(results['Premium Shoppers (Segment 1)'].values())
    
    # Find common parameters
    common_params = []
    common_value_values = []
    common_premium_values = []
    
    for i, param in enumerate(value_params):
        if any(p in param for p in ['top_dept_spend', 'unique_products']):
            # These parameters are in both segments
            common_params.append(param.replace('_multiplier', ''))
            common_value_values.append(value_values[i])
            
            # Find matching parameter in premium
            for j, p in enumerate(premium_params):
                if param == p:
                    common_premium_values.append(premium_values[j])
                    break
    
    # Plot common parameters
    x = np.arange(len(common_params))
    width = 0.35
    
    ax.bar(x - width/2, common_value_values, width, label='Value Shoppers')
    ax.bar(x + width/2, common_premium_values, width, label='Premium Shoppers')
    
    # Add labels and formatting
    ax.set_ylabel('Multiplier Value')
    ax.set_title('Optimal Parameter Values by Segment')
    ax.set_xticks(x)
    ax.set_xticklabels(common_params)
    ax.legend()
    
    # Add horizontal line at 1.0 (no change)
    ax.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(common_value_values):
        ax.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    for i, v in enumerate(common_premium_values):
        ax.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'segment_parameter_comparison.png'))
    
    # Create a visualization of segment-specific parameters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Value Shoppers unique parameters
    value_unique = [p for p in value_params if p not in premium_params or p in ['top_dept_spend_multiplier', 'unique_products_multiplier']]
    value_unique_values = [value_values[i] for i, p in enumerate(value_params) if p in value_unique]
    value_unique_labels = [p.replace('_multiplier', '') for p in value_unique]
    
    # Premium Shoppers unique parameters
    premium_unique = [p for p in premium_params if p not in value_params or p in ['top_dept_spend_multiplier', 'unique_products_multiplier']]
    premium_unique_values = [premium_values[i] for i, p in enumerate(premium_params) if p in premium_unique]
    premium_unique_labels = [p.replace('_multiplier', '') for p in premium_unique]
    
    # Plot Value Shoppers
    ax1.barh(value_unique_labels, value_unique_values, color='blue', alpha=0.7)
    ax1.set_title('Value Shoppers (Segment 0)')
    ax1.axvline(x=1.0, color='r', linestyle='-', alpha=0.3)
    ax1.set_xlim(0.5, 2.0)
    
    # Add value labels
    for i, v in enumerate(value_unique_values):
        ax1.text(v + 0.02, i, f'{v:.2f}', va='center')
    
    # Plot Premium Shoppers
    ax2.barh(premium_unique_labels, premium_unique_values, color='green', alpha=0.7)
    ax2.set_title('Premium Shoppers (Segment 1)')
    ax2.axvline(x=1.0, color='r', linestyle='-', alpha=0.3)
    ax2.set_xlim(0.5, 2.0)
    
    # Add value labels
    for i, v in enumerate(premium_unique_values):
        ax2.text(v + 0.02, i, f'{v:.2f}', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'segment_specific_parameters.png'))
    
    plt.close('all')

if __name__ == "__main__":
    main() 