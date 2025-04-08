import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import json
import os
from sklearn.inspection import permutation_importance

# Create results directory
os.makedirs('variable_optimization_results', exist_ok=True)

def main():
    print("=" * 60)
    print("Transition Variable Prediction Model")
    print("=" * 60)
    
    # 1. Load customer features
    print("\nLoading customer features data...")
    try:
        features_df = pd.read_csv("analysis_results/customer_features.csv")
        print(f"Loaded {len(features_df)} customer records with {len(features_df.columns)} features")
    except FileNotFoundError:
        print("Error: Customer features data not found. Please make sure the file exists.")
        return
    
    # 2. Load optimal transition matrix
    try:
        with open("optimization_results/optimization_results.json", "r") as f:
            optimization_results = json.load(f)
            optimal_transitions = optimization_results["optimal_transition_matrix"]
            current_transitions = optimization_results["current_transition_matrix"]
    except FileNotFoundError:
        print("Optimization results not found. Using default values...")
        # Use the output from our previous optimization
        optimal_transitions = {
            "Premium": {"Premium": 1.0, "Value": 0.0, "Inactive": 0.0},
            "Value": {"Premium": 0.338, "Value": 0.662, "Inactive": 0.0},
            "Inactive": {"Premium": 0.302, "Value": 0.026, "Inactive": 0.671}
        }
        current_transitions = {
            "Premium": {"Premium": 0.960, "Value": 0.038, "Inactive": 0.002},
            "Value": {"Premium": 0.038, "Value": 0.876, "Inactive": 0.019},
            "Inactive": {"Premium": 0.002, "Value": 0.059, "Inactive": 0.971}
        }
    
    # 3. Calculate key transition improvements
    key_transitions = [
        # Format: (from_state, to_state, improvement_priority)
        ("Value", "Premium", 1),    # Highest priority: upgrading Value customers
        ("Inactive", "Premium", 2), # Second priority: reactivating to Premium
        ("Premium", "Premium", 3),  # Third priority: retaining Premium customers
        ("Inactive", "Value", 4)    # Fourth priority: reactivating to Value
    ]
    
    # Print transition targets
    print("\nKey Transition Targets:")
    print("From State | To State | Current | Target | Change")
    print("-----------|----------|---------|--------|-------")
    for from_state, to_state, _ in key_transitions:
        current = current_transitions[from_state][to_state]
        target = optimal_transitions[from_state][to_state]
        change = target - current
        print(f"{from_state:10} | {to_state:8} | {current:.4f} | {target:.4f} | {change:+.4f}")
    
    # 4. Prepare features and identify actionable variables
    print("\nPreparing features and identifying actionable variables...")
    
    # Pre-process the dataframe to handle missing values and categorical features
    features_df = preprocess_features(features_df)
    
    # Identify actionable features (these would be the ones marketers can influence)
    # For now, we'll use a predefined list based on business knowledge
    actionable_features = identify_actionable_features(features_df)
    
    print(f"Identified {len(actionable_features)} actionable features:")
    for i, feature in enumerate(actionable_features):
        print(f"  {i+1}. {feature}")
    
    # 5. Build transition calculation features
    print("\nCreating transition-based features...")
    # In a real implementation, we would add features that capture transition probabilities
    # For now, we'll simulate this by assuming existing features already capture this information
    
    # 6. Build inverse prediction models
    print("\nBuilding inverse prediction models for key transitions...")
    
    # Dictionary to store models for each transition
    transition_models = {}
    feature_importances = {}
    
    # For each key transition, build a model
    for from_state, to_state, priority in key_transitions:
        print(f"\nModeling {from_state} → {to_state} transition (Priority {priority}):")
        
        # In a real implementation, we would:
        # 1. Calculate actual transition probabilities for each customer
        # 2. Use these as target variables
        # 3. Train models to predict transitions from customer features
        
        # For demonstration, we'll generate synthetic transition data
        # based on customer features
        
        # Create synthetic transition probabilities
        transition_col = f"transition_{from_state}_to_{to_state}"
        features_df[transition_col] = generate_synthetic_transitions(
            features_df, from_state, to_state
        )
        
        # Select features and target
        X = features_df[actionable_features]
        y = features_df[transition_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create a pipeline with imputer and model
        print("Training gradient boosting model with imputation...")
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                               max_depth=3, random_state=42))
        ])
        
        # Train model pipeline
        pipeline.fit(X_train, y_train)
        
        # Extract the model from the pipeline
        model = pipeline.named_steps['model']
        
        # Evaluate model
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Model performance: MSE = {mse:.4f}, R² = {r2:.4f}")
        
        # Store model and pipeline
        transition_models[(from_state, to_state)] = {
            'pipeline': pipeline,
            'model': model
        }
        
        # Calculate feature importances
        # For permutation importance, we need to use the full pipeline
        importance = permutation_importance(
            pipeline, X_test, y_test, n_repeats=10, random_state=42
        )
        
        # Store and display feature importances
        feature_importances[(from_state, to_state)] = {
            'mean': importance.importances_mean,
            'std': importance.importances_std,
            'features': actionable_features
        }
        
        # Print top features
        indices = np.argsort(importance.importances_mean)[::-1]
        print("\nTop 5 features for this transition:")
        for i in range(min(5, len(actionable_features))):
            idx = indices[i]
            print(f"  {actionable_features[idx]}: {importance.importances_mean[idx]:.4f} ± {importance.importances_std[idx]:.4f}")
        
        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importances for {from_state} → {to_state} Transition")
        plt.barh(
            range(len(indices[:10])),
            importance.importances_mean[indices[:10]],
            yerr=importance.importances_std[indices[:10]],
            align="center",
        )
        plt.yticks(range(len(indices[:10])), [actionable_features[i] for i in indices[:10]])
        plt.xlabel("Relative Importance")
        plt.tight_layout()
        plt.savefig(f"variable_optimization_results/feature_importance_{from_state}_to_{to_state}.png")
        plt.close()
    
    # 7. Inverse prediction: What variable values would achieve target transitions?
    print("\nInverse prediction: determining optimal variable values...")
    
    # Get current average values for actionable features
    # Use imputed values to avoid NaN issues
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(features_df[actionable_features]),
        columns=actionable_features
    )
    current_values = X_imputed.mean()
    
    # Store predicted optimal values for each transition
    optimal_values = {}
    
    for from_state, to_state, priority in key_transitions:
        transition_key = (from_state, to_state)
        
        target_transition = optimal_transitions[from_state][to_state]
        current_transition = current_transitions[from_state][to_state]
        
        print(f"\nFor {from_state} → {to_state} transition:")
        print(f"  Current probability: {current_transition:.4f}")
        print(f"  Target probability: {target_transition:.4f}")
        
        # Get model and feature importances for this transition
        model_info = transition_models[transition_key]
        pipeline = model_info['pipeline']
        model = model_info['model']
        importances = feature_importances[transition_key]['mean']
        
        # Get the most important features for this transition
        top_indices = np.argsort(importances)[::-1][:5]  # Top 5 features
        top_features = [actionable_features[i] for i in top_indices]
        
        # Predict optimal values for these features
        # In a real implementation, we would use more sophisticated inverse prediction methods
        # For now, we'll use a simple approximation based on feature importance and direction
        optimal_values[transition_key] = predict_optimal_values(
            pipeline, features_df, actionable_features, current_values,
            top_indices, target_transition, current_transition
        )
        
        # Display results
        print("\nPredicted optimal values for key variables:")
        print("Feature | Current Value | Optimal Value | % Change")
        print("--------|---------------|---------------|--------")
        
        for i, idx in enumerate(top_indices):
            feature = actionable_features[idx]
            current = current_values[feature]
            optimal = optimal_values[transition_key][feature]
            pct_change = (optimal / current - 1) * 100
            
            print(f"{feature:20} | {current:13.4f} | {optimal:13.4f} | {pct_change:+.2f}%")
    
    # 8. Generate marketing strategy recommendations
    print("\nGenerating marketing strategy recommendations...")
    
    # Aggregate optimal values across transitions, weighted by priority
    aggregated_optimal_values = aggregate_optimal_values(
        optimal_values, key_transitions, actionable_features
    )
    
    # Calculate overall changes needed
    changes = []
    for feature in actionable_features:
        current = current_values[feature]
        optimal = aggregated_optimal_values[feature]
        pct_change = (optimal / current - 1) * 100
        changes.append((feature, current, optimal, pct_change))
    
    # Sort by absolute percentage change
    changes.sort(key=lambda x: abs(x[3]), reverse=True)
    
    # Display top changes
    print("\nTop recommended variable changes:")
    print("Feature | Current Value | Optimal Value | % Change")
    print("--------|---------------|---------------|--------")
    
    for feature, current, optimal, pct_change in changes[:10]:
        print(f"{feature:20} | {current:13.4f} | {optimal:13.4f} | {pct_change:+.2f}%")
    
    # Create summary visualizations
    plot_variable_changes(changes[:15])
    
    # 9. Save results
    save_results(
        actionable_features, current_values, aggregated_optimal_values,
        optimal_transitions, current_transitions, feature_importances
    )
    
    print("\nAnalysis complete! Results saved to variable_optimization_results/")


def preprocess_features(df):
    """
    Preprocess features to handle missing values and categorical variables
    """
    # Create a copy to avoid modifying original
    processed_df = df.copy()
    
    # Handle categorical variables
    # For now, we'll just drop them to keep things simple for this example
    categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
    
    # Print categorical columns that will be excluded
    if len(categorical_cols) > 0:
        print(f"Excluding {len(categorical_cols)} categorical columns: {list(categorical_cols)}")
    
    # Keep only numeric columns
    processed_df = processed_df.select_dtypes(include=['int64', 'float64'])
    
    # Fill missing values with median for numeric columns
    # We won't actually modify the dataframe here since we'll use a pipeline with imputer
    # But we'll report the number of missing values
    missing_counts = processed_df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]
    
    if len(cols_with_missing) > 0:
        print(f"Found {len(cols_with_missing)} columns with missing values:")
        for col, count in cols_with_missing.items():
            print(f"  {col}: {count} missing values")
    
    return processed_df


def identify_actionable_features(df):
    """
    Identify features that marketers can directly influence
    """
    # In a real implementation, this would be based on domain knowledge
    # and feature metadata. For now, we'll use a predefined list.
    
    all_features = df.columns.tolist()
    
    # Filter for actionable features (this is a simplified approach)
    actionable_features = [
        col for col in all_features if any(term in col.lower() for term in [
            'discount', 'coupon', 'campaign', 'promotion', 'email', 'frequency',
            'basket', 'spend', 'visit', 'purchase', 'category', 'price',
            'redemption', 'loyalty', 'channel', 'marketing', 'dept', 'total'
        ])
    ]
    
    # If we don't find enough actionable features, use some default ones
    if len(actionable_features) < 5:
        print("Warning: Not enough actionable features identified. Using synthetic features.")
        # Create some synthetic features for demonstration
        synthetic_features = [
            'discount_rate',
            'campaign_frequency',
            'coupon_value',
            'email_frequency',
            'premium_product_ratio',
            'basket_size_avg',
            'promotion_exposure',
            'loyalty_points_per_dollar',
            'category_diversity_index',
            'price_sensitivity'
        ]
        
        # Add these columns with random values if they don't exist
        for feature in synthetic_features:
            if feature not in df.columns:
                df[feature] = np.random.normal(0.5, 0.2, size=len(df))
                actionable_features.append(feature)
    
    return actionable_features


def generate_synthetic_transitions(df, from_state, to_state):
    """
    Generate synthetic transition probabilities based on customer features
    This is a placeholder for actual transition probability calculations
    """
    # In a real implementation, we would calculate actual transition probs
    # from historical data. For now, we'll generate synthetic values.
    
    n_samples = len(df)
    
    # Base probabilities for different transitions
    if from_state == "Premium" and to_state == "Premium":
        # Premium retention: naturally high
        base_prob = 0.9
        noise_scale = 0.05
    elif from_state == "Value" and to_state == "Premium":
        # Value to Premium upgrade: moderate
        base_prob = 0.05
        noise_scale = 0.1
    elif from_state == "Inactive" and to_state == "Premium":
        # Inactive to Premium: low
        base_prob = 0.01
        noise_scale = 0.02
    elif from_state == "Inactive" and to_state == "Value":
        # Inactive to Value: low-moderate
        base_prob = 0.03
        noise_scale = 0.05
    else:
        # Other transitions
        base_prob = 0.1
        noise_scale = 0.1
    
    # Generate random probabilities around the base
    probabilities = np.random.normal(base_prob, noise_scale, size=n_samples)
    
    # Ensure values are between 0 and 1
    probabilities = np.clip(probabilities, 0, 1)
    
    # Make probabilities correlate with some features
    if 'discount_rate' in df.columns:
        probabilities += 0.2 * df['discount_rate']
    
    if 'campaign_frequency' in df.columns:
        probabilities += 0.15 * df['campaign_frequency']
        
    # Renormalize to 0-1 range
    probabilities = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min())
    
    return probabilities


def predict_optimal_values(pipeline, df, feature_names, current_values, 
                          top_indices, target_prob, current_prob):
    """
    Predict optimal values for actionable features to achieve target transition probability
    
    In a real implementation, this would use more sophisticated inverse prediction methods
    For now, we'll use a simple approximation
    """
    # Get the direction of change needed
    direction = 1 if target_prob > current_prob else -1
    
    # Calculate the magnitude of change needed
    magnitude = abs(target_prob - current_prob) / current_prob
    scaling_factor = min(1.0, magnitude * 2)  # Cap at 100% change
    
    # Create a copy of current values to modify
    optimal_values = current_values.copy()
    
    # Adjust the most important features proportionally to their importance
    for i, idx in enumerate(top_indices):
        feature = feature_names[idx]
        
        # Weight change by feature importance rank (first feature gets largest change)
        importance_weight = 1.0 / (i + 1)
        
        # Calculate adjustment (larger for more important features)
        adjustment = direction * scaling_factor * importance_weight
        
        # Apply adjustment
        optimal_values[feature] *= (1 + adjustment)
        
        # Ensure values stay in reasonable ranges
        # This is a simplified approach; actual implementation would use feature-specific bounds
        feature_min = df[feature].min()
        feature_max = df[feature].max()
        feature_mean = df[feature].mean()
        
        # Handle potential NaN values
        if pd.isna(feature_min) or pd.isna(feature_max) or pd.isna(feature_mean):
            # Use non-NaN values only for min/max calculation
            non_nan_values = df[feature].dropna()
            if len(non_nan_values) > 0:
                feature_min = non_nan_values.min()
                feature_max = non_nan_values.max()
                feature_mean = non_nan_values.mean()
            else:
                # If all values are NaN, use default bounds
                feature_min = 0.01
                feature_max = 1.0
                feature_mean = 0.5
        
        lower_bound = max(0.01, feature_min)
        upper_bound = min(feature_max * 1.5, feature_mean * 3)
        
        optimal_values[feature] = np.clip(optimal_values[feature], lower_bound, upper_bound)
    
    return optimal_values


def aggregate_optimal_values(optimal_values, key_transitions, feature_names):
    """
    Aggregate optimal values across transitions, weighted by priority
    """
    # Initialize with zeros
    aggregated = {feature: 0 for feature in feature_names}
    
    # Calculate priority weights (higher priority = higher weight)
    priorities = [transition[2] for transition in key_transitions]
    max_priority = max(priorities)
    weights = [max_priority - p + 1 for p in priorities]  # Reverse priorities
    weight_sum = sum(weights)
    normalized_weights = [w / weight_sum for w in weights]
    
    # Weighted average of optimal values
    for i, (from_state, to_state, _) in enumerate(key_transitions):
        transition_key = (from_state, to_state)
        if transition_key in optimal_values:
            for feature in feature_names:
                if feature in optimal_values[transition_key]:
                    aggregated[feature] += (
                        optimal_values[transition_key][feature] * normalized_weights[i]
                    )
    
    return aggregated


def plot_variable_changes(changes):
    """
    Create visualizations of recommended variable changes
    """
    features = [c[0] for c in changes]
    current_values = [c[1] for c in changes]
    optimal_values = [c[2] for c in changes]
    pct_changes = [c[3] for c in changes]
    
    # Shorten feature names for better display
    short_features = [f[:15] + '...' if len(f) > 15 else f for f in features]
    
    # Plot percentage changes
    plt.figure(figsize=(12, 8))
    colors = ['green' if pc > 0 else 'red' for pc in pct_changes]
    plt.barh(short_features, pct_changes, color=colors)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.title('Recommended Variable Changes (%)')
    plt.xlabel('Percentage Change')
    plt.tight_layout()
    plt.savefig('variable_optimization_results/recommended_changes_pct.png')
    plt.close()
    
    # Plot current vs optimal values
    plt.figure(figsize=(12, 8))
    x = range(len(features))
    width = 0.35
    
    plt.barh([i - width/2 for i in x], current_values, width, label='Current', color='blue', alpha=0.6)
    plt.barh([i + width/2 for i in x], optimal_values, width, label='Optimal', color='orange', alpha=0.6)
    
    plt.yticks(x, short_features)
    plt.title('Current vs. Optimal Variable Values')
    plt.legend()
    plt.tight_layout()
    plt.savefig('variable_optimization_results/current_vs_optimal.png')
    plt.close()


def save_results(features, current_values, optimal_values, 
                optimal_transitions, current_transitions, feature_importances):
    """
    Save analysis results to files
    """
    # Create a summary dataframe
    results_df = pd.DataFrame({
        'Feature': features,
        'Current_Value': [current_values[f] for f in features],
        'Optimal_Value': [optimal_values[f] for f in features],
        'Percentage_Change': [(optimal_values[f] / current_values[f] - 1) * 100 for f in features]
    })
    
    # Save to CSV
    results_df.to_csv('variable_optimization_results/optimal_variable_values.csv', index=False)
    
    # Save full results as JSON
    full_results = {
        'features': features,
        'current_values': {f: float(current_values[f]) for f in features},
        'optimal_values': {f: float(optimal_values[f]) for f in features},
        'percentage_changes': {f: float((optimal_values[f] / current_values[f] - 1) * 100) for f in features},
        'current_transitions': current_transitions,
        'optimal_transitions': optimal_transitions,
        'feature_importances': {
            f"{k[0]}_to_{k[1]}": {
                'features': v['features'],
                'importance_means': v['mean'].tolist(),
                'importance_stds': v['std'].tolist()
            }
            for k, v in feature_importances.items()
        }
    }
    
    with open('variable_optimization_results/full_results.json', 'w') as f:
        json.dump(full_results, f, indent=4)


if __name__ == "__main__":
    main() 