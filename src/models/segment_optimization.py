#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Segment Optimization Analysis
Identifying key drivers of spend within each customer segment
and developing targeted strategies to increase customer value
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Set file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'optimization_results')
SEGMENTATION_DIR = os.path.join(BASE_DIR, 'segmentation_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_segmented_data():
    """Load the segmented customer data"""
    segment_file = os.path.join(SEGMENTATION_DIR, 'customer_segments.csv')
    
    if os.path.exists(segment_file):
        print("Loading segmented customer data...")
        segmented_df = pd.read_csv(segment_file)
        print(f"Loaded data with {len(segmented_df)} customers and {segmented_df.shape[1]} features.")
        return segmented_df
    else:
        print("Error: Segmented customer data file not found.")
        return None

def analyze_segment_correlations(segmented_df):
    """Analyze correlations between variables for each segment"""
    print("Analyzing segment-specific correlations...")
    
    # Identify clusters
    clusters = segmented_df['cluster'].unique()
    n_clusters = len(clusters)
    
    # Target variable
    target_var = 'total_spend'
    
    # Select numeric features for correlation analysis
    numeric_cols = segmented_df.select_dtypes(include=['int64', 'float64']).columns
    
    # Drop identifier columns and target
    exclude_cols = ['household_key', 'cluster', 'hierarchical_cluster']
    feature_cols = [col for col in numeric_cols if col != target_var and col not in exclude_cols]
    
    # Calculate and store correlations for each segment
    segment_correlations = {}
    top_correlations = {}
    
    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 6*n_clusters))
    if n_clusters == 1:
        axes = [axes]
    
    for i, cluster in enumerate(clusters):
        # Subset data for this cluster
        cluster_data = segmented_df[segmented_df['cluster'] == cluster]
        
        # Calculate correlations with target
        correlations = cluster_data[feature_cols + [target_var]].corr()[target_var].drop(target_var)
        segment_correlations[cluster] = correlations
        
        # Sort and get top positive and negative correlations
        sorted_corr = correlations.sort_values(ascending=False)
        top_pos = sorted_corr.head(10)
        top_neg = sorted_corr.tail(5)
        top_correlations[cluster] = pd.concat([top_pos, top_neg])
        
        # Plot
        sns.barplot(x=top_correlations[cluster].values, y=top_correlations[cluster].index, ax=axes[i])
        axes[i].set_title(f'Cluster {cluster}: Top Correlations with Total Spend')
        axes[i].set_xlabel('Correlation Coefficient')
        axes[i].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'segment_correlations.png'))
    
    # Save correlation data
    correlation_df = pd.DataFrame({f'Cluster_{cluster}': corr for cluster, corr in segment_correlations.items()})
    correlation_df.to_csv(os.path.join(RESULTS_DIR, 'segment_correlations.csv'))
    
    return segment_correlations, top_correlations

def build_segment_regression_models(segmented_df):
    """Build regression models for each segment to identify coefficients"""
    print("Building segment-specific regression models...")
    
    # Identify clusters
    clusters = segmented_df['cluster'].unique()
    
    # Target variable
    target_var = 'total_spend'
    
    # Select numeric features for correlation analysis
    numeric_cols = segmented_df.select_dtypes(include=['int64', 'float64']).columns
    
    # Drop identifier columns and target
    exclude_cols = ['household_key', 'cluster', 'hierarchical_cluster', target_var]
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Store model results
    model_results = {}
    
    for cluster in clusters:
        # Subset data for this cluster
        cluster_data = segmented_df[segmented_df['cluster'] == cluster].copy()
        
        print(f"\nBuilding regression model for Cluster {cluster}...")
        
        # Handle missing values
        for col in feature_cols:
            if cluster_data[col].isnull().any():
                cluster_data[col].fillna(cluster_data[col].median(), inplace=True)
        
        # Prepare features and target
        X = cluster_data[feature_cols]
        y = cluster_data[target_var]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 1. Linear regression with statsmodels (for coefficients and p-values)
        X_train_sm = sm.add_constant(X_train)
        sm_model = sm.OLS(y_train, X_train_sm).fit()
        
        # 2. Random Forest (for feature importances)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Evaluate models
        X_test_sm = sm.add_constant(X_test)
        sm_preds = sm_model.predict(X_test_sm)
        rf_preds = rf_model.predict(X_test)
        
        sm_r2 = r2_score(y_test, sm_preds)
        rf_r2 = r2_score(y_test, rf_preds)
        
        print(f"Linear model R²: {sm_r2:.4f}")
        print(f"Random Forest R²: {rf_r2:.4f}")
        
        # Extract coefficients from linear model
        coef_df = pd.DataFrame({
            'Feature': sm_model.params.index,
            'Coefficient': sm_model.params.values,
            'StdError': sm_model.bse.values,
            'P-Value': sm_model.pvalues.values,
            'Significant': sm_model.pvalues.values < 0.05
        })
        coef_df = coef_df[coef_df['Feature'] != 'const'].sort_values('Coefficient', ascending=False)
        
        # Extract feature importances from RF model
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Store results
        model_results[cluster] = {
            'linear_model': sm_model,
            'rf_model': rf_model,
            'coefficients': coef_df,
            'importances': importance_df,
            'linear_r2': sm_r2,
            'rf_r2': rf_r2
        }
        
        # Save coefficients to file
        coef_df.to_csv(os.path.join(RESULTS_DIR, f'cluster_{cluster}_coefficients.csv'), index=False)
        importance_df.to_csv(os.path.join(RESULTS_DIR, f'cluster_{cluster}_importances.csv'), index=False)
        
        # Plot significant coefficients
        sig_coef = coef_df[coef_df['Significant']].sort_values('Coefficient', ascending=False)
        
        if len(sig_coef) > 0:
            plt.figure(figsize=(12, len(sig_coef) * 0.4 + 2))
            colors = ['green' if c > 0 else 'red' for c in sig_coef['Coefficient']]
            
            plt.barh(sig_coef['Feature'], sig_coef['Coefficient'], color=colors)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.title(f'Cluster {cluster}: Significant Coefficients for Total Spend')
            plt.xlabel('Standardized Coefficient')
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f'cluster_{cluster}_significant_coefficients.png'))
        
        # Plot top 15 feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
        plt.title(f'Cluster {cluster}: Top 15 Features for Predicting Total Spend')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'cluster_{cluster}_feature_importances.png'))
    
    # Save summary of results
    with open(os.path.join(RESULTS_DIR, 'regression_model_summary.txt'), 'w') as f:
        for cluster, results in model_results.items():
            f.write(f"==== Cluster {cluster} Regression Results ====\n\n")
            f.write(f"Linear model R²: {results['linear_r2']:.4f}\n")
            f.write(f"Random Forest R²: {results['rf_r2']:.4f}\n\n")
            
            f.write("Top 10 coefficients (linear model):\n")
            for _, row in results['coefficients'].head(10).iterrows():
                f.write(f"{row['Feature']}: {row['Coefficient']:.4f} (p-value: {row['P-Value']:.4f})\n")
            
            f.write("\nTop 10 feature importances (random forest):\n")
            for _, row in results['importances'].head(10).iterrows():
                f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
            
            f.write("\n\n")
    
    return model_results

def identify_actionable_insights(segment_correlations, model_results):
    """Identify actionable insights and optimization strategies for each segment"""
    print("Identifying actionable insights for each segment...")
    
    # Identify clusters
    clusters = list(segment_correlations.keys())
    
    # Define categories of variables
    variable_categories = {
        'shopping_behavior': ['total_baskets', 'basket_frequency', 'active_weeks', 'unique_stores_visited', 'day_span'],
        'product_diversity': ['unique_products', 'product_diversity'],
        'campaign_engagement': ['campaigns_participated', 'coupons_redeemed', 'coupon_redemption_rate'],
        'basket_composition': ['avg_basket_value', 'avg_items_per_basket', 'total_items'],
        'discounts': ['total_retail_discount', 'total_coupon_discount', 'total_coupon_match_discount', 'discount_ratio']
    }
    
    # Store optimization strategies
    segment_strategies = {}
    
    for cluster in clusters:
        # Get correlation data for this cluster
        correlations = segment_correlations[cluster]
        
        # Get regression coefficients
        coefficients = model_results[cluster]['coefficients']
        
        # Identify key drivers with significant positive coefficients
        positive_drivers = coefficients[
            (coefficients['Coefficient'] > 0) & 
            (coefficients['Significant']) & 
            (coefficients['Feature'] != 'const')
        ].sort_values('Coefficient', ascending=False)
        
        # Store categories of positive drivers
        positive_driver_categories = set()
        for _, row in positive_drivers.iterrows():
            feature = row['Feature']
            for category, vars in variable_categories.items():
                if feature in vars:
                    positive_driver_categories.add(category)
        
        # Develop strategies based on positive drivers
        strategies = []
        
        if 'shopping_behavior' in positive_driver_categories:
            shop_drivers = [f for f in positive_drivers['Feature'] if f in variable_categories['shopping_behavior']]
            if 'basket_frequency' in shop_drivers or 'total_baskets' in shop_drivers:
                strategies.append("Increase visit frequency through time-limited offers and reminders")
            if 'active_weeks' in shop_drivers:
                strategies.append("Encourage regular shopping patterns with weekly specials")
            if 'unique_stores_visited' in shop_drivers:
                strategies.append("Promote cross-store shopping with multi-location incentives")
        
        if 'product_diversity' in positive_driver_categories:
            strategies.append("Encourage exploration of new product categories with personalized recommendations")
            strategies.append("Introduce category-specific promotions to expand product variety")
        
        if 'campaign_engagement' in positive_driver_categories:
            strategies.append("Increase participation in targeted campaigns with personalized offers")
            if 'coupon_redemption_rate' in positive_drivers['Feature'].values:
                strategies.append("Enhance coupon redemption rates with more relevant and timely offers")
        
        if 'basket_composition' in positive_driver_categories:
            if 'avg_basket_value' in positive_drivers['Feature'].values:
                strategies.append("Increase average transaction value with bundled offers and premium products")
            if 'avg_items_per_basket' in positive_drivers['Feature'].values:
                strategies.append("Encourage larger baskets with quantity discounts")
        
        if 'discounts' in positive_driver_categories:
            strategies.append("Optimize discount strategy to drive incremental purchases")
            disc_drivers = [f for f in positive_drivers['Feature'] if f in variable_categories['discounts']]
            if 'total_coupon_discount' in disc_drivers or 'total_coupon_match_discount' in disc_drivers:
                strategies.append("Focus on personalized coupon offers matched to customer preferences")
        
        # Additional strategies based on segment characteristics
        if cluster == 0:  # Value segment (low-value, low-frequency)
            strategies.append("Focus on increasing shopping frequency for Value Shoppers")
            strategies.append("Introduce lower barrier-to-entry loyalty programs")
            strategies.append("Use loss-leader pricing on key value items to drive traffic")
        else:  # Premium segment (high-value, high-frequency)
            strategies.append("Emphasize premium products and exclusive offerings for Premium Shoppers")
            strategies.append("Develop VIP benefits and experiential rewards")
            strategies.append("Create cross-category bundle offers that increase average basket value")
        
        # Store strategies
        segment_strategies[cluster] = strategies
    
    # Save strategies to file
    with open(os.path.join(RESULTS_DIR, 'segment_optimization_strategies.txt'), 'w') as f:
        for cluster, strategies in segment_strategies.items():
            f.write(f"==== Optimization Strategies for Cluster {cluster} ====\n\n")
            for i, strategy in enumerate(strategies, 1):
                f.write(f"{i}. {strategy}\n")
            f.write("\n\n")
    
    return segment_strategies

def simulate_spending_improvement(segmented_df, model_results, segment_strategies):
    """Simulate potential spending improvements based on optimization strategies"""
    print("Simulating potential spending improvements...")
    
    # Identify clusters
    clusters = segmented_df['cluster'].unique()
    
    # Current average spend by segment
    current_spend = segmented_df.groupby('cluster')['total_spend'].mean()
    
    # Simulate improvements
    improvement_scenarios = {}
    
    for cluster in clusters:
        # Get the model for this cluster
        rf_model = model_results[cluster]['rf_model']
        
        # Get a sample of customers from this cluster
        cluster_data = segmented_df[segmented_df['cluster'] == cluster].copy()
        
        # Define optimization scenarios based on strategies
        scenarios = []
        
        if cluster == 0:  # Value Shoppers
            # Scenario 1: Frequency improvement
            scenarios.append({
                'name': 'Increased Shopping Frequency',
                'changes': {
                    'basket_frequency': lambda x: x * 1.2,  # 20% increase in frequency
                    'total_baskets': lambda x: x * 1.2,     # 20% increase in baskets
                    'active_weeks': lambda x: min(x * 1.15, 104)  # 15% increase in active weeks
                }
            })
            
            # Scenario 2: Product diversity
            scenarios.append({
                'name': 'Increased Product Diversity',
                'changes': {
                    'unique_products': lambda x: x * 1.15,  # 15% increase in product variety
                    'product_diversity': lambda x: min(x * 1.1, 1.0)  # 10% increase in diversity
                }
            })
            
            # Scenario 3: Coupon utilization
            scenarios.append({
                'name': 'Enhanced Coupon Utilization',
                'changes': {
                    'coupons_redeemed': lambda x: x + 1,  # +1 redeemed coupons
                    'coupon_redemption_rate': lambda x: min(x + 0.1, 1.0),  # +10% redemption rate
                    'total_coupon_discount': lambda x: x * 1.25  # 25% increase in coupon discount
                }
            })
            
        else:  # Premium Shoppers
            # Scenario 1: Premium offers
            scenarios.append({
                'name': 'Premium Product Offers',
                'changes': {
                    'avg_basket_value': lambda x: x * 1.1,  # 10% increase in average basket value
                    'avg_items_per_basket': lambda x: x * 1.05  # 5% increase in items per basket
                }
            })
            
            # Scenario 2: Cross-category shopping
            scenarios.append({
                'name': 'Cross-Category Shopping',
                'changes': {
                    'unique_products': lambda x: x * 1.1,  # 10% increase in unique products
                    'unique_stores_visited': lambda x: min(x + 1, 10)  # Visit 1 more store
                }
            })
            
            # Scenario 3: Loyalty enhancement
            scenarios.append({
                'name': 'Enhanced Loyalty & Engagement',
                'changes': {
                    'active_weeks': lambda x: min(x * 1.05, 104),  # 5% more active weeks
                    'campaigns_participated': lambda x: x * 1.15,  # 15% more campaign participation
                    'basket_frequency': lambda x: x * 1.05  # 5% increase in frequency
                }
            })
        
        # Combined scenario
        combined_changes = {}
        for scenario in scenarios:
            combined_changes.update(scenario['changes'])
        
        scenarios.append({
            'name': 'All Strategies Combined',
            'changes': combined_changes
        })
        
        # Run simulations
        scenario_results = []
        feature_cols = [col for col in rf_model.feature_names_in_ if col not in ['household_key', 'cluster', 'hierarchical_cluster', 'total_spend']]
        
        for scenario in scenarios:
            # Create a copy of the data for this scenario
            scenario_data = cluster_data.copy()
            
            # Apply changes
            for feature, change_func in scenario['changes'].items():
                if feature in scenario_data.columns:
                    scenario_data[feature] = scenario_data[feature].apply(change_func)
            
            # Predict new spend
            X_scenario = scenario_data[feature_cols]
            predicted_spend = rf_model.predict(X_scenario)
            
            # Calculate improvement
            current_avg = cluster_data['total_spend'].mean()
            new_avg = predicted_spend.mean()
            improvement = new_avg - current_avg
            pct_improvement = (improvement / current_avg) * 100
            
            # Store results
            scenario_results.append({
                'scenario': scenario['name'],
                'current_spend': current_avg,
                'predicted_spend': new_avg,
                'improvement': improvement,
                'improvement_pct': pct_improvement
            })
        
        # Store all scenario results for this cluster
        improvement_scenarios[cluster] = scenario_results
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    # Set up the plot
    cluster_names = [f"Cluster {c}" for c in clusters]
    scenario_names = [s['scenario'] for s in improvement_scenarios[clusters[0]]]
    
    # Set width of bars
    bar_width = 0.15
    positions = np.arange(len(cluster_names))
    
    # Plot bars for each scenario
    for i, scenario in enumerate(scenario_names):
        improvements = [improvement_scenarios[c][i]['improvement_pct'] for c in clusters]
        plt.bar(positions + i*bar_width, improvements, bar_width, label=scenario)
    
    # Add labels and legend
    plt.xlabel('Customer Segment')
    plt.ylabel('Predicted Spend Improvement (%)')
    plt.title('Potential Spend Improvement by Segment and Strategy')
    plt.xticks(positions + bar_width * (len(scenario_names) - 1) / 2, cluster_names)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(RESULTS_DIR, 'spend_improvement_scenarios.png'))
    
    # Save detailed results to file
    with open(os.path.join(RESULTS_DIR, 'spend_improvement_analysis.txt'), 'w') as f:
        for cluster, scenarios in improvement_scenarios.items():
            f.write(f"==== Cluster {cluster} Improvement Scenarios ====\n\n")
            f.write(f"Current average spend: ${current_spend[cluster]:.2f}\n\n")
            
            for scenario in scenarios:
                f.write(f"Scenario: {scenario['scenario']}\n")
                f.write(f"  Predicted average spend: ${scenario['predicted_spend']:.2f}\n")
                f.write(f"  Improvement: ${scenario['improvement']:.2f}\n")
                f.write(f"  Improvement percentage: {scenario['improvement_pct']:.2f}%\n\n")
            
            f.write("\n")
    
    return improvement_scenarios

def create_segment_strategy_flow(segment_strategies):
    """Create a flowchart for segment-based customer strategy"""
    print("Creating segment strategy flowchart...")
    
    # Extract strategies for each segment
    value_strategies = segment_strategies.get(0, [])
    premium_strategies = segment_strategies.get(1, [])
    
    # Create a figure
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath
    
    fig = Figure(figsize=(10, 12))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    
    # Turn off axis
    ax.axis('off')
    
    # Define node positions
    nodes = {
        'start': (5, 11),
        'new_customer': (5, 9),
        'segment_check': (5, 7),
        'value_segment': (2, 5),
        'premium_segment': (8, 5),
        'value_strategy1': (2, 3),
        'value_strategy2': (2, 1),
        'premium_strategy1': (8, 3),
        'premium_strategy2': (8, 1)
    }
    
    # Define node styles
    node_styles = {
        'start': {'fc': 'lightgreen', 'ec': 'green'},
        'new_customer': {'fc': 'lightskyblue', 'ec': 'blue'},
        'segment_check': {'fc': 'lightyellow', 'ec': 'goldenrod'},
        'value_segment': {'fc': 'lightcoral', 'ec': 'red'},
        'premium_segment': {'fc': 'mediumpurple', 'ec': 'darkviolet'},
        'value_strategy1': {'fc': 'mistyrose', 'ec': 'lightcoral'},
        'value_strategy2': {'fc': 'mistyrose', 'ec': 'lightcoral'},
        'premium_strategy1': {'fc': 'lavender', 'ec': 'mediumpurple'},
        'premium_strategy2': {'fc': 'lavender', 'ec': 'mediumpurple'}
    }
    
    # Draw nodes
    for node, (x, y) in nodes.items():
        style = node_styles.get(node, {'fc': 'white', 'ec': 'black'})
        ax.add_patch(mpatches.Rectangle((x-2, y-0.5), 4, 1, fc=style['fc'], ec=style['ec'], alpha=0.8))
    
    # Draw arrows
    ax.arrow(5, 10.5, 0, -0.9, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax.arrow(5, 8.5, 0, -0.9, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax.arrow(4.8, 6.5, -2.3, -0.9, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax.arrow(5.2, 6.5, 2.3, -0.9, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax.arrow(2, 4.5, 0, -0.9, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax.arrow(2, 2.5, 0, -0.9, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax.arrow(8, 4.5, 0, -0.9, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax.arrow(8, 2.5, 0, -0.9, head_width=0.2, head_length=0.2, fc='black', ec='black')
    
    # Add node text
    ax.text(5, 11, "Start", ha='center', va='center', fontweight='bold')
    ax.text(5, 9, "New Customer Acquisition", ha='center', va='center', fontweight='bold')
    ax.text(5, 7, "Predict Customer Segment\nUsing Random Forest Model", ha='center', va='center', fontweight='bold')
    ax.text(2, 5, "Value Shopper Segment", ha='center', va='center', fontweight='bold')
    ax.text(8, 5, "Premium Shopper Segment", ha='center', va='center', fontweight='bold')
    
    # Get top 2 strategies for each segment
    value_top2 = value_strategies[:2] if len(value_strategies) >= 2 else value_strategies + ["Strategy placeholder"]
    premium_top2 = premium_strategies[:2] if len(premium_strategies) >= 2 else premium_strategies + ["Strategy placeholder"]
    
    # Add strategy text (wrap long texts)
    ax.text(2, 3, _wrap_text(value_top2[0], 25), ha='center', va='center')
    if len(value_top2) > 1:
        ax.text(2, 1, _wrap_text(value_top2[1], 25), ha='center', va='center')
    
    ax.text(8, 3, _wrap_text(premium_top2[0], 25), ha='center', va='center')
    if len(premium_top2) > 1:
        ax.text(8, 1, _wrap_text(premium_top2[1], 25), ha='center', va='center')
    
    # Add decision labels
    ax.text(3.8, 6.7, "Low Value\nLow Frequency", ha='center', va='center', fontsize=8)
    ax.text(6.2, 6.7, "High Value\nHigh Frequency", ha='center', va='center', fontsize=8)
    
    # Save figure
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'segment_strategy_flow.png'))
    
    return fig

def _wrap_text(text, width):
    """Helper function to wrap text to a specific width"""
    import textwrap
    return '\n'.join(textwrap.wrap(text, width=width))

def main():
    """Main function for segment optimization analysis"""
    print("Starting segment optimization analysis...")
    
    # Load segmented customer data
    segmented_df = load_segmented_data()
    
    if segmented_df is None:
        print("Error: Could not load segmented customer data.")
        return
    
    # Analyze segment-specific correlations
    segment_correlations, top_correlations = analyze_segment_correlations(segmented_df)
    
    # Build regression models for each segment
    model_results = build_segment_regression_models(segmented_df)
    
    # Identify actionable insights
    segment_strategies = identify_actionable_insights(segment_correlations, model_results)
    
    # Simulate spending improvements
    improvement_scenarios = simulate_spending_improvement(segmented_df, model_results, segment_strategies)
    
    # Create segment strategy flow
    create_segment_strategy_flow(segment_strategies)
    
    print("\n===== SEGMENT OPTIMIZATION SUMMARY =====")
    
    for cluster in segmented_df['cluster'].unique():
        print(f"\nCluster {cluster} Key Insights:")
        
        # Top correlations
        print("  Top positive correlations with spend:")
        top_pos = top_correlations[cluster].sort_values(ascending=False).head(3)
        for feature, corr in top_pos.items():
            print(f"    - {feature}: {corr:.4f}")
        
        # Key strategies
        print("  Recommended optimization strategies:")
        for i, strategy in enumerate(segment_strategies[cluster][:3], 1):
            print(f"    {i}. {strategy}")
        
        # Potential improvements
        combined_scenario = improvement_scenarios[cluster][-1]  # Get the combined scenario
        print(f"  Potential spend improvement with combined strategies: {combined_scenario['improvement_pct']:.2f}%")
        print(f"    Current avg. spend: ${combined_scenario['current_spend']:.2f}")
        print(f"    Potential avg. spend: ${combined_scenario['predicted_spend']:.2f}")
    
    print("\nAnalysis completed successfully. Results saved to the 'optimization_results' directory.")

if __name__ == "__main__":
    main() 