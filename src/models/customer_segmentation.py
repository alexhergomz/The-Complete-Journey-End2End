#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Customer Segmentation Analysis for Dunnhumby The Complete Journey dataset
Using PCA for dimension reduction followed by K-means and hierarchical clustering
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Set file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'dunnhumby.db')
RESULTS_DIR = os.path.join(BASE_DIR, 'segmentation_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create a connection to the database
def connect_to_db():
    """Create a connection to the SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        print("Connected to the database successfully!")
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

# Execute SQL query and return results as a pandas DataFrame
def execute_query(conn, query):
    """Execute a SQL query and return results as a pandas DataFrame"""
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error executing query: {e}")
        return None

def load_customer_features():
    """Load customer features from previous analysis or recreate if needed"""
    feature_file = os.path.join(BASE_DIR, 'analysis_results', 'customer_features.csv')
    
    if os.path.exists(feature_file):
        print("Loading existing customer features dataset...")
        customer_features = pd.read_csv(feature_file)
        print(f"Loaded feature dataset with {len(customer_features)} customers and {customer_features.shape[1]} features.")
        return customer_features
    else:
        print("Customer features file not found. Creating new feature dataset...")
        # Import the function from our previous analysis
        from customer_purchase_analysis import create_customer_features
        customer_features = create_customer_features()
        return customer_features

def prepare_features_for_clustering(customer_features):
    """Prepare customer features for clustering analysis"""
    print("Preparing features for clustering...")
    
    # Make a copy to avoid modifying the original
    df = customer_features.copy()
    
    # Handle missing values for numerical features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Select features for clustering
    # Based on previous correlation analysis, these were significant features
    clustering_features = [
        'total_baskets', 'unique_products', 'total_items', 'total_spend',
        'avg_transaction_value', 'day_span', 'active_weeks', 'total_retail_discount',
        'total_coupon_discount', 'total_coupon_match_discount', 'unique_stores_visited',
        'top_dept_spend', 'campaigns_participated', 'coupons_redeemed', 'basket_frequency'
    ]
    
    # Only keep features that exist in the dataset
    clustering_features = [f for f in clustering_features if f in df.columns]
    
    # Create a subset with only the features for clustering
    clustering_df = df[['household_key'] + clustering_features]
    
    print(f"Selected {len(clustering_features)} features for clustering analysis.")
    
    return clustering_df, df

def perform_pca_analysis(clustering_df):
    """Perform PCA to reduce dimensions before clustering"""
    print("Performing PCA analysis...")
    
    # Remove identifier columns
    features_df = clustering_df.drop('household_key', axis=1)
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_df)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    # Get explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, color='blue', label='Individual explained variance')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
    plt.axhline(y=0.8, linestyle='--', color='red', alpha=0.5, label='80% Threshold')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pca_explained_variance.png'))
    
    # Determine optimal number of components (80% variance explained)
    n_components = np.argmax(cumulative_variance >= 0.8) + 1
    print(f"Optimal number of principal components: {n_components} (explains {cumulative_variance[n_components-1]:.2%} of variance)")
    
    # Create reduced PCA with optimal components
    pca_optimal = PCA(n_components=n_components)
    pca_optimal_result = pca_optimal.fit_transform(scaled_data)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(
        data=pca_optimal_result,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    pca_df['household_key'] = clustering_df['household_key'].values
    
    # Create component loadings dataframe for interpretation
    loadings = pd.DataFrame(
        data=pca_optimal.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=features_df.columns
    )
    
    # Save loadings to file
    loadings.to_csv(os.path.join(RESULTS_DIR, 'pca_component_loadings.csv'))
    
    # Visualization of first 2 principal components
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.3)
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
    plt.title('PCA: First Two Principal Components')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pca_first_two_components.png'))
    
    # Print top contributing features for each principal component
    print("\nTop contributing features for each principal component:")
    for i in range(n_components):
        pc_loadings = loadings[f'PC{i+1}'].abs().sort_values(ascending=False)
        print(f"\nPC{i+1} top features:")
        print(pc_loadings.head(5))
    
    return pca_df, loadings, pca_optimal, scaled_data

def determine_optimal_clusters(pca_df):
    """Determine the optimal number of clusters for K-means"""
    print("Determining optimal number of clusters...")
    
    # Remove identifier column for clustering
    pca_data = pca_df.drop('household_key', axis=1)
    
    # Calculate silhouette scores and inertia for different numbers of clusters
    silhouette_scores = []
    ch_scores = []  # Calinski-Harabasz score
    inertia = []
    
    max_clusters = min(10, len(pca_data) - 1)  # Maximum 10 clusters or n-1 (whichever is smaller)
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pca_data)
        
        silhouette_avg = silhouette_score(pca_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        ch_score = calinski_harabasz_score(pca_data, cluster_labels)
        ch_scores.append(ch_score)
        
        inertia.append(kmeans.inertia_)
    
    # Plot silhouette scores
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average Silhouette Score')
    plt.title('Silhouette Score by Cluster Count')
    
    # Plot Elbow curve for inertia
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), inertia, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method: Inertia by Cluster Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'optimal_cluster_analysis.png'))
    
    # Determine optimal number of clusters
    # Using silhouette score (higher is better)
    optimal_silhouette = np.argmax(silhouette_scores) + 2  # +2 because we start from 2 clusters
    
    # Using elbow method (look for the "elbow" point)
    # We'll use the point where the rate of decrease in inertia slows down
    inertia_diffs = np.diff(inertia)
    inertia_diffs2 = np.diff(inertia_diffs)
    elbow_point = np.argmax(inertia_diffs2) + 2  # +2 because we start from 2 and lose one in the diff
    
    print(f"Optimal clusters by silhouette score: {optimal_silhouette}")
    print(f"Optimal clusters by elbow method: {elbow_point}")
    
    # We'll return both, but favor silhouette score for the final decision
    return optimal_silhouette, elbow_point

def perform_kmeans_clustering(pca_df, n_clusters):
    """Perform K-means clustering on PCA-reduced data"""
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    
    # Remove identifier column for clustering
    pca_data = pca_df.drop('household_key', axis=1)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pca_data)
    
    # Add cluster labels to the dataframe
    pca_df['cluster'] = cluster_labels
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(pca_data, cluster_labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    
    # Visualize clusters (first 2 principal components)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=pca_df.columns[0],
        y=pca_df.columns[1],
        hue='cluster',
        data=pca_df,
        palette='viridis',
        alpha=0.7,
        legend='full'
    )
    plt.title(f'K-means Clustering with {n_clusters} Clusters')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'kmeans_clusters_{n_clusters}.png'))
    
    # Create 3D interactive plot if we have at least 3 principal components
    if pca_df.shape[1] >= 4:  # 3 PCs + household_key + cluster
        fig = px.scatter_3d(
            pca_df,
            x=pca_df.columns[0],
            y=pca_df.columns[1],
            z=pca_df.columns[2],
            color='cluster',
            opacity=0.7,
            title=f'K-means Clustering with {n_clusters} Clusters (3D)'
        )
        fig.write_html(os.path.join(RESULTS_DIR, f'kmeans_clusters_3d_{n_clusters}.html'))
    
    return pca_df, kmeans

def perform_hierarchical_clustering(pca_df, n_clusters):
    """Perform hierarchical clustering on PCA-reduced data"""
    print(f"Performing hierarchical clustering with {n_clusters} clusters...")
    
    # Remove identifier column for clustering
    pca_data = pca_df.drop(['household_key', 'cluster'], axis=1)
    
    # Compute linkage matrix
    Z = linkage(pca_data, method='ward')
    
    # Plot dendrogram (truncated for clarity if many customers)
    plt.figure(figsize=(12, 8))
    plt.title(f'Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('Sample index or (cluster size)')
    plt.ylabel('Distance')
    
    # Draw dendrogram with truncation if needed
    if len(pca_data) > 100:
        # Truncate the dendrogram to show only main branches
        dendrogram(
            Z,
            truncate_mode='lastp',  # show only the last p merged clusters
            p=20,  # show only the last 20 merged clusters
            leaf_rotation=90.,
            leaf_font_size=8.,
            show_contracted=True,
        )
    else:
        dendrogram(
            Z,
            leaf_rotation=90.,
            leaf_font_size=8.,
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'hierarchical_dendrogram_{n_clusters}.png'))
    
    # Cut the dendrogram to get n_clusters
    hierarchical_clusters = fcluster(Z, n_clusters, criterion='maxclust')
    
    # Add hierarchical cluster labels to the dataframe
    pca_df['hierarchical_cluster'] = hierarchical_clusters
    
    # Visualize hierarchical clusters (first 2 principal components)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=pca_df.columns[0],
        y=pca_df.columns[1],
        hue='hierarchical_cluster',
        data=pca_df,
        palette='Set1',
        alpha=0.7,
        legend='full'
    )
    plt.title(f'Hierarchical Clustering with {n_clusters} Clusters')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'hierarchical_clusters_{n_clusters}.png'))
    
    # Create 3D interactive plot if we have at least 3 principal components
    if pca_data.shape[1] >= 3:
        fig = px.scatter_3d(
            pca_df,
            x=pca_df.columns[0],
            y=pca_df.columns[1],
            z=pca_df.columns[2],
            color='hierarchical_cluster',
            opacity=0.7,
            title=f'Hierarchical Clustering with {n_clusters} Clusters (3D)'
        )
        fig.write_html(os.path.join(RESULTS_DIR, f'hierarchical_clusters_3d_{n_clusters}.html'))
    
    return pca_df, Z

def compare_clustering_methods(pca_df):
    """Compare K-means and hierarchical clustering results"""
    print("Comparing clustering methods...")
    
    # Create a cross-tabulation of K-means vs hierarchical clusters
    cluster_comparison = pd.crosstab(
        pca_df['cluster'],
        pca_df['hierarchical_cluster'],
        rownames=['K-means'],
        colnames=['Hierarchical']
    )
    
    # Save the comparison to file
    cluster_comparison.to_csv(os.path.join(RESULTS_DIR, 'cluster_comparison.csv'))
    
    # Visualize the comparison
    plt.figure(figsize=(10, 8))
    sns.heatmap(cluster_comparison, annot=True, cmap='YlGnBu', fmt='d')
    plt.title('Comparison of K-means and Hierarchical Clustering')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'cluster_comparison.png'))
    
    # Calculate agreement percentage
    total_customers = len(pca_df)
    
    # Calculate adjusted labels for maximum agreement
    # (This handles the fact that cluster numbering may differ between methods)
    agreements = 0
    for km_cluster in range(cluster_comparison.shape[0]):
        max_agreement = cluster_comparison.iloc[km_cluster].max()
        agreements += max_agreement
    
    agreement_percentage = (agreements / total_customers) * 100
    print(f"Agreement between clustering methods: {agreement_percentage:.2f}%")
    
    return cluster_comparison, agreement_percentage

def analyze_clusters(pca_df, full_df, kmeans_model, pca_model, scaler, n_clusters):
    """Analyze the characteristics of each cluster"""
    print("Analyzing cluster characteristics...")
    
    # Merge clustering results with full data
    clustered_df = full_df.copy()
    cluster_info = pca_df[['household_key', 'cluster', 'hierarchical_cluster']]
    clustered_df = pd.merge(clustered_df, cluster_info, on='household_key')
    
    # Calculate summary statistics for each K-means cluster
    kmeans_cluster_profile = clustered_df.groupby('cluster').agg({
        'total_baskets': 'mean',
        'unique_products': 'mean',
        'total_items': 'mean',
        'total_spend': 'mean',
        'avg_transaction_value': 'mean',
        'active_weeks': 'mean',
        'campaigns_participated': 'mean',
        'coupons_redeemed': 'mean',
        'basket_frequency': 'mean',
        'household_key': 'count'
    }).rename(columns={'household_key': 'customer_count'})
    
    # Save cluster profiles
    kmeans_cluster_profile.to_csv(os.path.join(RESULTS_DIR, f'kmeans_cluster_profile_{n_clusters}.csv'))
    
    # Calculate summary statistics for each hierarchical cluster
    hierarchical_cluster_profile = clustered_df.groupby('hierarchical_cluster').agg({
        'total_baskets': 'mean',
        'unique_products': 'mean',
        'total_items': 'mean',
        'total_spend': 'mean',
        'avg_transaction_value': 'mean',
        'active_weeks': 'mean',
        'campaigns_participated': 'mean',
        'coupons_redeemed': 'mean',
        'basket_frequency': 'mean',
        'household_key': 'count'
    }).rename(columns={'household_key': 'customer_count'})
    
    # Save cluster profiles
    hierarchical_cluster_profile.to_csv(os.path.join(RESULTS_DIR, f'hierarchical_cluster_profile_{n_clusters}.csv'))
    
    # Visualize key metrics by cluster
    key_metrics = ['total_spend', 'total_baskets', 'unique_products', 'basket_frequency']
    
    # K-means cluster visualization
    plt.figure(figsize=(12, 10))
    for i, metric in enumerate(key_metrics):
        plt.subplot(2, 2, i+1)
        sns.barplot(x=kmeans_cluster_profile.index, y=kmeans_cluster_profile[metric], palette='viridis')
        plt.title(f'Average {metric.replace("_", " ").title()} by K-means Cluster')
        plt.xlabel('Cluster')
        plt.ylabel(metric.replace('_', ' ').title())
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'kmeans_cluster_metrics_{n_clusters}.png'))
    
    # Add descriptive names to clusters based on characteristics
    cluster_names = []
    for i in range(n_clusters):
        profile = kmeans_cluster_profile.iloc[i]
        
        # Determine spending level
        if profile['total_spend'] > kmeans_cluster_profile['total_spend'].mean() * 1.5:
            spend_level = "High-Spenders"
        elif profile['total_spend'] > kmeans_cluster_profile['total_spend'].mean() * 0.75:
            spend_level = "Regular-Spenders"
        else:
            spend_level = "Low-Spenders"
        
        # Determine frequency level
        if profile['basket_frequency'] > kmeans_cluster_profile['basket_frequency'].mean() * 1.5:
            freq_level = "Frequent-Shoppers"
        elif profile['basket_frequency'] > kmeans_cluster_profile['basket_frequency'].mean() * 0.75:
            freq_level = "Regular-Shoppers"
        else:
            freq_level = "Infrequent-Shoppers"
        
        # Determine product diversity
        if profile['unique_products'] > kmeans_cluster_profile['unique_products'].mean() * 1.5:
            diversity = "High-Variety"
        elif profile['unique_products'] > kmeans_cluster_profile['unique_products'].mean() * 0.75:
            diversity = "Medium-Variety"
        else:
            diversity = "Low-Variety"
        
        cluster_name = f"Cluster {i}: {spend_level}, {freq_level}, {diversity}"
        cluster_names.append(cluster_name)
    
    # Save cluster names
    with open(os.path.join(RESULTS_DIR, f'kmeans_cluster_names_{n_clusters}.txt'), 'w') as f:
        for i, name in enumerate(cluster_names):
            f.write(f"Cluster {i}: {name}\n")
            f.write(f"Count: {kmeans_cluster_profile.iloc[i]['customer_count']}\n")
            f.write(f"Avg Spend: ${kmeans_cluster_profile.iloc[i]['total_spend']:.2f}\n")
            f.write(f"Avg Baskets: {kmeans_cluster_profile.iloc[i]['total_baskets']:.1f}\n")
            f.write(f"Avg Products: {kmeans_cluster_profile.iloc[i]['unique_products']:.1f}\n")
            f.write(f"Shopping Frequency: {kmeans_cluster_profile.iloc[i]['basket_frequency']:.3f}\n")
            f.write("-" * 50 + "\n")
    
    # Print cluster names
    print("\nK-means Cluster Descriptions:")
    for name in cluster_names:
        print(name)
    
    # Create a radar chart to compare clusters
    categories = ['Spending', 'Frequency', 'Products', 'Avg Basket', 'Campaigns', 'Coupons']
    
    # Create normalized values for radar chart
    radar_df = kmeans_cluster_profile[['total_spend', 'basket_frequency', 'unique_products', 
                                      'avg_transaction_value', 'campaigns_participated', 'coupons_redeemed']]
    
    # Normalize values to 0-1 scale for comparison
    radar_df = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())
    
    # Rename columns for the chart
    radar_df.columns = categories
    
    # Create a plotly radar chart
    fig = go.Figure()
    
    for i in range(n_clusters):
        values = radar_df.iloc[i].tolist()
        values.append(values[0])  # Close the loop
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],  # Close the loop
            fill='toself',
            name=f'Cluster {i}'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=f"Customer Segment Comparison"
    )
    
    fig.write_html(os.path.join(RESULTS_DIR, f'cluster_radar_chart_{n_clusters}.html'))
    
    return clustered_df, kmeans_cluster_profile, cluster_names

def train_segment_classifier(clustered_df, pca_model, scaler, n_clusters):
    """Train a classifier to predict customer segments"""
    print("Training segment prediction model...")
    
    # Prepare features and target variable
    X = clustered_df.drop(['household_key', 'cluster', 'hierarchical_cluster'], axis=1)
    
    # Select only numeric features for now
    X = X.select_dtypes(include=['int64', 'float64'])
    
    # Fill any missing values
    X = X.fillna(X.median())
    
    y = clustered_df['cluster']  # Use K-means clusters as target
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = clf.predict(X_test)
    
    # Print classification report
    print("\nSegment Prediction Model Performance:")
    print(classification_report(y_test, y_pred))
    
    # Save classification report
    with open(os.path.join(RESULTS_DIR, f'segment_classifier_report_{n_clusters}.txt'), 'w') as f:
        f.write("Segment Prediction Model Performance:\n")
        f.write(classification_report(y_test, y_pred))
    
    # Get feature importances
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': clf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Save feature importances
    feature_importances.to_csv(os.path.join(RESULTS_DIR, f'segment_feature_importances_{n_clusters}.csv'), index=False)
    
    # Plot top 15 feature importances
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(15))
    plt.title('Top 15 Features for Segment Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'segment_feature_importances_{n_clusters}.png'))
    
    print(f"Top 5 features for segment prediction:")
    for i, row in feature_importances.head(5).iterrows():
        print(f"  - {row['Feature']}: {row['Importance']:.4f}")
    
    return clf, feature_importances

def main():
    """Main function for customer segmentation analysis"""
    print("Starting customer segmentation analysis...")
    
    # Load customer features
    customer_features = load_customer_features()
    
    if customer_features is None:
        print("Error: Could not load customer features.")
        return
    
    # Prepare features for clustering
    clustering_df, full_df = prepare_features_for_clustering(customer_features)
    
    # Perform PCA analysis
    pca_df, loadings, pca_model, scaled_data = perform_pca_analysis(clustering_df)
    
    # Determine optimal number of clusters
    silhouette_optimal, elbow_optimal = determine_optimal_clusters(pca_df)
    
    # Use the optimal number of clusters (favoring silhouette)
    # But if they differ a lot, use the smaller number for interpretability
    if abs(silhouette_optimal - elbow_optimal) > 2:
        n_clusters = min(silhouette_optimal, elbow_optimal)
    else:
        n_clusters = silhouette_optimal
    
    print(f"Using {n_clusters} clusters for final analysis")
    
    # Perform K-means clustering
    pca_df, kmeans_model = perform_kmeans_clustering(pca_df, n_clusters)
    
    # Perform hierarchical clustering
    pca_df, linkage_matrix = perform_hierarchical_clustering(pca_df, n_clusters)
    
    # Compare clustering methods
    cluster_comparison, agreement = compare_clustering_methods(pca_df)
    
    # Analyze clusters
    clustered_df, cluster_profile, cluster_names = analyze_clusters(
        pca_df, full_df, kmeans_model, pca_model, scaled_data, n_clusters
    )
    
    # Train segment classifier
    classifier, feature_importances = train_segment_classifier(
        clustered_df, pca_model, scaled_data, n_clusters
    )
    
    # Save final segmented dataset
    clustered_df.to_csv(os.path.join(RESULTS_DIR, 'customer_segments.csv'), index=False)
    
    print("\n===== SEGMENTATION ANALYSIS SUMMARY =====")
    print(f"Total customers analyzed: {len(clustered_df)}")
    print(f"Number of clusters identified: {n_clusters}")
    print(f"Agreement between K-means and hierarchical clustering: {agreement:.2f}%")
    print("\nSegment descriptions:")
    for name in cluster_names:
        print(f"  - {name}")
    
    print("\nPotential applications:")
    print("1. Personalized marketing campaigns for each segment")
    print("2. Segment-based product recommendations")
    print("3. Customer acquisition strategies targeting look-alike customers")
    print("4. Retention strategies tailored to high-value segments")
    print("5. Store layout optimization for different customer segments")
    
    print("\nAnalysis completed successfully. Results saved to the 'segmentation_results' directory.")

if __name__ == "__main__":
    main() 