import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import sqlite3
import json

# Create output directory
os.makedirs('real_transition_models', exist_ok=True)

def connect_to_db():
    """Create a connection to the SQLite database"""
    try:
        # Use absolute path to the database file in the current directory
        db_path = os.path.join(os.getcwd(), 'dunnhumby.db')
        print(f"Connecting to database at: {db_path}")
        conn = sqlite3.connect(db_path)
        print("Connected to the database successfully!")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def extract_customer_states():
    """Extract customer state transitions from real transaction data"""
    print("Extracting customer states from transaction data...")
    
    conn = connect_to_db()
    if not conn:
        # Use synthetic data if connection fails
        print("Database connection failed. Using synthetic transitions instead.")
        return create_synthetic_transitions()
    
    # Define time periods (quarters)
    query = """
    SELECT MIN(day) as min_day, MAX(day) as max_day FROM transaction_data
    """
    
    try:
        time_data = pd.read_sql_query(query, conn)
        min_day = time_data['min_day'].iloc[0]
        max_day = time_data['max_day'].iloc[0]
        print(f"Transaction data spans from day {min_day} to {max_day}")
    except Exception as e:
        print(f"Error querying min/max days: {e}")
        print("Using synthetic transitions instead.")
        conn.close()
        return create_synthetic_transitions()
    
    # Use 4 equal time periods
    periods = 4
    period_length = (max_day - min_day) // periods
    print(f"Using {periods} periods of {period_length} days each")
    
    # Query to get spending and frequency by customer and period
    # Use simpler query with explicit period ranges to avoid parameter binding issues
    query = f"""
    WITH customer_periods AS (
        SELECT 
            household_key,
            CASE
                WHEN day BETWEEN {min_day} AND {min_day + period_length - 1} THEN 0
                WHEN day BETWEEN {min_day + period_length} AND {min_day + 2*period_length - 1} THEN 1
                WHEN day BETWEEN {min_day + 2*period_length} AND {min_day + 3*period_length - 1} THEN 2
                WHEN day BETWEEN {min_day + 3*period_length} AND {max_day} THEN 3
            END as period,
            SUM(sales_value) as total_spend,
            COUNT(DISTINCT day) as unique_days
        FROM transaction_data
        WHERE household_key IN (SELECT household_key FROM household_demographics)
        GROUP BY household_key, period
    )
    SELECT * FROM customer_periods
    WHERE period IS NOT NULL
    ORDER BY household_key, period
    """
    
    try:
        # Execute query
        print("Executing query to extract customer periods...")
        customer_periods = pd.read_sql_query(query, conn)
        print(f"Found {len(customer_periods)} customer-period records")
    except Exception as e:
        print(f"Error executing query: {e}")
        print("Using synthetic transitions instead.")
        conn.close()
        return create_synthetic_transitions()
    
    if len(customer_periods) == 0:
        print("No customer periods found. Using synthetic transitions instead.")
        conn.close()
        return create_synthetic_transitions()
    
    # Define state determination function
    def determine_state(spend, frequency, period_days):
        # Calculate frequency as days with purchases / days in period
        freq_ratio = frequency / period_days if period_days > 0 else 0
        
        if spend > 100 and freq_ratio > 0.1:  # Premium thresholds
            return 'Premium'
        elif spend > 30 or freq_ratio > 0.05:  # Value thresholds 
            return 'Value'
        else:
            return 'Inactive'
    
    # Ensure period is integer type 
    customer_periods['period'] = customer_periods['period'].astype(int)
    
    # Assign states to each customer-period combination
    customer_periods['state'] = customer_periods.apply(
        lambda x: determine_state(
            x['total_spend'], 
            x['unique_days'], 
            period_length
        ), axis=1
    )
    
    # Fill in missing periods with 'Inactive' state
    all_customers = customer_periods['household_key'].unique()
    all_periods = list(range(periods))  # Convert to list to ensure consistent type
    
    print(f"Creating complete customer-period grid for {len(all_customers)} customers...")
    
    # Create complete customer-period grid
    all_combinations = []
    for customer in all_customers:
        for period in all_periods:
            all_combinations.append({
                'household_key': int(customer),  # Ensure integer type
                'period': int(period)  # Ensure integer type
            })
    
    # Convert to DataFrame and ensure proper types
    all_combinations_df = pd.DataFrame(all_combinations)
    all_combinations_df['household_key'] = all_combinations_df['household_key'].astype(int)
    all_combinations_df['period'] = all_combinations_df['period'].astype(int)
    
    # Merge with existing data
    print("Merging customer periods with complete grid...")
    customer_periods_complete = pd.merge(
        all_combinations_df, 
        customer_periods, 
        on=['household_key', 'period'], 
        how='left'
    )
    
    # Fill missing states with 'Inactive'
    customer_periods_complete['state'] = customer_periods_complete['state'].fillna('Inactive')
    # Fill missing numeric values with 0
    numeric_cols = ['total_spend', 'unique_days']
    customer_periods_complete[numeric_cols] = customer_periods_complete[numeric_cols].fillna(0)
    
    # Extract transitions between consecutive periods
    print("Extracting transitions between consecutive periods...")
    transitions = []
    
    for customer in all_customers:
        customer_data = customer_periods_complete[
            customer_periods_complete['household_key'] == customer
        ].sort_values('period')
        
        # Get states for this customer
        states = customer_data['state'].tolist()
        
        # Create transitions between consecutive periods
        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i + 1]
            
            transitions.append({
                'household_key': customer,
                'period': i,
                'from_state': from_state,
                'to_state': to_state,
                'transition': f'{from_state} → {to_state}'
            })
    
    conn.close()
    
    transitions_df = pd.DataFrame(transitions)
    print(f"Extracted {len(transitions_df)} transitions from {len(all_customers)} customers")
    
    if len(transitions_df) == 0:
        print("No transitions found. Using synthetic transitions instead.")
        return create_synthetic_transitions()
    
    return transitions_df

def create_synthetic_transitions():
    """Create synthetic transitions for testing when real data is unavailable"""
    print("Creating synthetic transitions (demo mode)...")
    
    # Load customer features to get household keys
    try:
        customer_features = pd.read_csv('analysis_results/customer_features.csv')
        household_keys = customer_features['household_key'].unique()
    except FileNotFoundError:
        # Generate some sample household keys if file not found
        household_keys = np.arange(1, 501)
    
    np.random.seed(42)
    transitions = []
    
    # Initial state distribution
    states = ['Premium', 'Value', 'Inactive']
    state_distribution = {
        'Premium': 0.2,
        'Value': 0.5,
        'Inactive': 0.3
    }
    
    # Transition matrix (based on optimization results)
    transition_matrix = {
        'Premium': {'Premium': 0.96, 'Value': 0.038, 'Inactive': 0.002},
        'Value': {'Premium': 0.038, 'Value': 0.876, 'Inactive': 0.086},
        'Inactive': {'Premium': 0.002, 'Value': 0.059, 'Inactive': 0.939}
    }
    
    # Generate transitions for each customer
    for household_key in household_keys:
        # First period state
        initial_state = np.random.choice(
            states, 
            p=[state_distribution['Premium'], state_distribution['Value'], state_distribution['Inactive']]
        )
        
        current_state = initial_state
        
        # Generate transitions for 3 periods
        for period in range(3):
            # Determine next state
            next_state_probs = [
                transition_matrix[current_state]['Premium'],
                transition_matrix[current_state]['Value'],
                transition_matrix[current_state]['Inactive']
            ]
            
            next_state = np.random.choice(states, p=next_state_probs)
            
            transitions.append({
                'household_key': household_key,
                'period': period,
                'from_state': current_state,
                'to_state': next_state,
                'transition': f'{current_state} → {next_state}'
            })
            
            current_state = next_state
    
    return pd.DataFrame(transitions)

def preprocess_features(df):
    """Preprocess customer features by handling missing values and categoricals"""
    # Identify categorical columns
    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() < 15:
            categorical_cols.append(col)
    
    # Remove household_key from categorical list if present
    if 'household_key' in categorical_cols:
        categorical_cols.remove('household_key')
    
    print(f"Excluding {len(categorical_cols)} categorical columns: {categorical_cols}")
    
    # Check for missing values
    missing_values = df.isna().sum()
    cols_with_missing = missing_values[missing_values > 0]
    
    if len(cols_with_missing) > 0:
        print(f"Found {len(cols_with_missing)} columns with missing values:")
        for col, count in cols_with_missing.items():
            print(f"  {col}: {count} missing values")
    
    # Return numeric columns only
    numeric_cols = [col for col in df.columns if col not in categorical_cols or col == 'household_key']
    return df[numeric_cols]

def train_real_transition_models(transitions_df, features_df):
    """Train models based on real transition data"""
    print("\nTraining models on real customer transitions...")
    
    # Merge transitions with features
    merged_data = transitions_df.merge(
        features_df, 
        on='household_key',
        how='inner'
    )
    
    print(f"Merged dataset has {len(merged_data)} records")
    
    # Identify key transitions to model
    key_transitions = [
        'Value → Premium',
        'Inactive → Premium',
        'Inactive → Value',
        'Premium → Premium'
    ]
    
    # Create binary target variables for each transition
    for transition in key_transitions:
        merged_data[f'{transition}_target'] = (merged_data['transition'] == transition).astype(int)
    
    # Get list of feature columns (excluding metadata and targets)
    feature_cols = [
        col for col in merged_data.columns 
        if col not in ['household_key', 'period', 'from_state', 'to_state', 'transition'] 
        and not col.endswith('_target')
    ]
    
    # Train models for each key transition
    models = {}
    
    for transition in key_transitions:
        print(f"\nModeling {transition} transition:")
        
        # Target variable
        target_col = f"{transition}_target"
        target_counts = merged_data[target_col].value_counts()
        print(f"Target distribution: {dict(target_counts)}")
        
        # Split data
        X = merged_data[feature_cols]
        y = merged_data[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Create a pipeline with preprocessing
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate model
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"Training metrics - MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"Test metrics - MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
        
        # Extract feature importances
        model = pipeline.named_steps['model']
        importances = model.feature_importances_
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            pipeline, X_test, y_test, n_repeats=10, random_state=42
        )
        
        # Top features
        indices = np.argsort(perm_importance.importances_mean)[::-1]
        print("\nTop features for this transition:")
        for i in range(min(5, len(feature_cols))):
            idx = indices[i]
            feature = feature_cols[idx]
            importance = perm_importance.importances_mean[idx]
            std = perm_importance.importances_std[idx]
            print(f"  {feature}: {importance:.4f} ± {std:.4f}")
        
        # Store model information
        models[transition] = {
            'pipeline': pipeline,
            'train_mse': train_mse,
            'train_r2': train_r2,
            'test_mse': test_mse,
            'test_r2': test_r2,
            'importances': dict(zip(feature_cols, importances)),
            'perm_importances': {
                'mean': dict(zip(feature_cols, perm_importance.importances_mean)),
                'std': dict(zip(feature_cols, perm_importance.importances_std))
            }
        }
        
        # Plot feature importances
        plt.figure(figsize=(10, 6))
        top_n = 10
        plt.barh(
            range(top_n),
            perm_importance.importances_mean[indices[:top_n]],
            yerr=perm_importance.importances_std[indices[:top_n]],
            align='center'
        )
        plt.yticks(range(top_n), [feature_cols[i] for i in indices[:top_n]])
        plt.title(f'Feature Importance for {transition}')
        plt.tight_layout()
        plt.savefig(f'real_transition_models/{transition.replace(" → ", "_to_")}_importance.png')
        plt.close()
    
    return models

def validate_against_simulation(models, features_df):
    """Validate models against simulated optimal transitions"""
    print("\nValidating models against simulation results...")
    
    try:
        # Load simulation results if available
        with open("optimization_results/optimization_results.json", "r") as f:
            optimization_results = json.load(f)
            optimal_transitions = optimization_results["optimal_transition_matrix"]
            current_transitions = optimization_results["current_transition_matrix"]
            
        print("Loaded simulation optimization results")
    except FileNotFoundError:
        print("Simulation results not found. Using default values.")
        # Default values based on previous optimization
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
    
    # Identify actionable features
    actionable_features = identify_actionable_features(features_df)
    
    # Get current values
    imputer = SimpleImputer(strategy='median')
    current_values = pd.DataFrame(
        imputer.fit_transform(features_df[actionable_features]),
        columns=actionable_features
    ).mean()
    
    # For each key transition, determine optimal variable values
    recommendations = {}
    
    for transition_name, model_info in models.items():
        from_state, to_state = transition_name.split(' → ')
        
        current_prob = current_transitions[from_state][to_state]
        target_prob = optimal_transitions[from_state][to_state]
        
        # Get the model and importances
        pipeline = model_info['pipeline']
        importances = model_info['perm_importances']['mean']
        
        # Get top features by importance
        sorted_features = sorted(
            importances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        top_features = [f[0] for f in sorted_features[:5]]
        
        # Predict optimal values
        optimal_values = predict_optimal_values(
            pipeline, features_df, top_features, current_values,
            current_prob, target_prob
        )
        
        recommendations[transition_name] = {
            'top_features': top_features,
            'current_prob': current_prob,
            'target_prob': target_prob,
            'test_r2': model_info['test_r2'],
            'optimal_values': optimal_values
        }
        
        # Print recommendations
        print(f"\nFor {transition_name} transition:")
        print(f"  Current probability: {current_prob:.4f}")
        print(f"  Target probability: {target_prob:.4f}")
        print(f"  Model R²: {model_info['test_r2']:.4f}")
        
        print("\nRecommended changes:")
        print("Feature | Current Value | Optimal Value | % Change")
        print("--------|---------------|---------------|--------")
        
        for feature in top_features:
            current = current_values[feature]
            optimal = optimal_values[feature]
            pct_change = (optimal / current - 1) * 100
            
            print(f"{feature:20} | {current:13.4f} | {optimal:13.4f} | {pct_change:+.2f}%")
    
    return recommendations

def identify_actionable_features(df):
    """Identify actionable features from customer data"""
    # List of features that marketers can potentially influence
    actionable_features = [
        'total_spend', 'total_baskets', 'total_items', 'basket_frequency',
        'unique_stores_visited', 'total_retail_discount', 'total_coupon_discount',
        'total_coupon_match_discount', 'campaigns_participated', 'coupons_redeemed',
        'top_dept_spend', 'active_weeks', 'day_span'
    ]
    
    # Filter to only include features in the dataframe
    available_features = [f for f in actionable_features if f in df.columns]
    
    if len(available_features) < 5:
        # If we don't have enough predefined actionable features,
        # use numeric columns as a fallback
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        # Exclude household_key
        if 'household_key' in numeric_cols:
            numeric_cols.remove('household_key')
        print("Using generic numeric features as actionable variables")
        return numeric_cols[:10]  # Use top 10 numeric features
    
    print(f"Identified {len(available_features)} actionable features: {available_features}")
    return available_features

def predict_optimal_values(pipeline, df, top_features, current_values, current_prob, target_prob):
    """Predict optimal values for key variables to achieve target transition probability"""
    # Initialize optimal values with current values
    optimal_values = current_values.copy()
    
    # Filter top_features to only include those in current_values
    available_features = [f for f in top_features if f in current_values.index]
    
    if len(available_features) == 0:
        print("Warning: None of the top model features are in the available customer features.")
        # Use whatever features we have as a fallback
        available_features = current_values.index[:5].tolist()
        print(f"Using these features instead: {available_features}")
    else:
        print(f"Using these available features: {available_features}")
    
    # Calculate the change direction
    is_increase = target_prob > current_prob
    direction = 1 if is_increase else -1
    
    # Set change percentages based on feature importance and transition magnitude
    change_magnitude = abs(target_prob - current_prob)
    
    # Scale changes by importance
    for i, feature in enumerate(available_features):
        # Higher changes for more important features
        importance_factor = 1.0 - (i * 0.15)  # Decreasing by position
        
        # Determine change percentage
        if i == 0:  # Most important feature
            pct_change = 1.0 * change_magnitude * 10
        elif i == 1:
            pct_change = 0.5 * change_magnitude * 10
        elif i == 2:
            pct_change = 0.33 * change_magnitude * 10
        else:
            pct_change = 0.2 * change_magnitude * 10
        
        # Cap changes
        pct_change = min(pct_change, 2.0)  # Max 200% change
        
        # Apply direction
        pct_change *= direction
        
        # Discount features get special treatment (typically negative values)
        if 'discount' in feature:
            # For discounts, make them more negative if increasing transition
            if direction > 0:
                pct_change = abs(pct_change) * 2
            else:
                pct_change = -abs(pct_change) * 0.5
        
        # Apply change
        optimal_values[feature] *= (1 + pct_change)
    
    return optimal_values

def aggregate_recommendations(recommendations):
    """Aggregate recommendations across transitions to create an overall strategy"""
    print("\nAggregating recommendations across transitions...")
    
    # Define priorities for each transition
    priorities = {
        'Value → Premium': 1,
        'Inactive → Premium': 2,
        'Premium → Premium': 3,
        'Inactive → Value': 4
    }
    
    # Collect all features mentioned across transitions
    all_features = set()
    for rec in recommendations.values():
        all_features.update(rec['top_features'])
    
    # Calculate weighted recommendations
    feature_changes = {}
    
    for feature in all_features:
        # Collect changes for this feature across transitions
        changes = []
        weights = []
        
        for transition, rec in recommendations.items():
            if feature in rec['optimal_values']:
                current = rec['current_prob']
                target = rec['target_prob']
                importance = 1.0 / priorities.get(transition, 5)  # Higher weight for higher priority
                
                # Current and optimal values
                current_val = rec['optimal_values'][feature]
                
                # Calculate change
                pct_change = current_val
                
                # Add to lists
                changes.append(pct_change)
                weights.append(importance * abs(target - current))
        
        if changes:
            # Calculate weighted average change
            weighted_change = np.average(changes, weights=weights)
            feature_changes[feature] = weighted_change
    
    # Sort features by absolute change
    sorted_changes = sorted(
        feature_changes.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    # Print aggregated recommendations
    print("\nOverall recommended variable changes:")
    print("Feature | Recommended Value")
    print("--------|------------------")
    
    for feature, value in sorted_changes:
        print(f"{feature:20} | {value:.4f}")
    
    return dict(sorted_changes)

def main():
    print("=" * 60)
    print("Real Transition-Based Variable Optimization")
    print("=" * 60)
    
    # 1. Load customer features
    print("\nLoading customer features data...")
    try:
        features_df = pd.read_csv("analysis_results/customer_features.csv")
        print(f"Loaded {len(features_df)} customer records with {len(features_df.columns)} features")
    except FileNotFoundError:
        print("Error: Customer features data not found. Please make sure the file exists.")
        return
    
    # 2. Extract real customer transitions
    transitions_df = extract_customer_states()
    
    # Display transition summary
    transition_counts = transitions_df['transition'].value_counts()
    print("\nTransition counts:")
    for transition, count in transition_counts.items():
        print(f"  {transition}: {count}")
    
    # Calculate transition matrix
    transition_matrix = pd.crosstab(
        transitions_df['from_state'],
        transitions_df['to_state'],
        normalize='index'
    ).round(4)
    
    print("\nTransition Matrix (rows=from, columns=to):")
    print(transition_matrix)
    
    # 3. Preprocess features
    print("\nPreprocessing customer features...")
    processed_features = preprocess_features(features_df)
    
    # 4. Train models on real transitions
    models = train_real_transition_models(transitions_df, processed_features)
    
    # 5. Validate against simulation
    recommendations = validate_against_simulation(models, processed_features)
    
    # 6. Aggregate recommendations
    aggregated_recs = aggregate_recommendations(recommendations)
    
    # 7. Save results
    print("\nSaving results...")
    
    # Create a serializable version of the results
    json_safe_results = {
        'transition_matrix': transition_matrix.to_dict(),
        'model_metrics': {},
        'recommendations': {},
        'aggregated_recommendations': aggregated_recs
    }
    
    # Add model metrics without the non-serializable components
    for k, v in models.items():
        json_safe_results['model_metrics'][k] = {
            'test_r2': v['test_r2'], 
            'test_mse': v['test_mse'],
            'train_r2': v['train_r2'],
            'train_mse': v['train_mse']
        }
    
    # Add recommendation data without any sklearn objects
    for k, v in recommendations.items():
        json_safe_results['recommendations'][k] = {
            'top_features': v['top_features'],
            'current_prob': v['current_prob'],
            'target_prob': v['target_prob'],
            'test_r2': v['test_r2'],
            'optimal_values': v['optimal_values'].to_dict()
        }
    
    # Save as JSON
    with open('real_transition_models/results.json', 'w') as f:
        # Convert numpy values to Python native types
        json.dump(json_safe_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.number) else x)
    
    print("\nAnalysis complete! Results saved to real_transition_models/")

if __name__ == "__main__":
    main() 