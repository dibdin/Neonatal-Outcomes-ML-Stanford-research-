#!/usr/bin/env python3
"""
Check for feature clusters via correlation heatmap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_and_process_data
import warnings
warnings.filterwarnings('ignore')

def check_feature_clusters():
    """Check for feature clusters via correlation analysis"""
    
    print("=== FEATURE CLUSTER ANALYSIS ===\n")
    
    # Test with biomarker data (most features)
    print("🔍 Analyzing biomarker features for clustering...")
    
    # Load biomarker data
    X, y, data_df = load_and_process_data(
        dataset_type='cord',
        model_type='biomarker',
        data_option=1,
        dropna=False,
        random_state=48,
        target_type='gestational_age',
        return_dataframe=True
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    
    # Calculate correlation matrix
    print("\n📊 Calculating correlation matrix...")
    corr_matrix = X.corr()
    
    # Find highly correlated feature pairs
    print("\n🔍 Finding highly correlated feature pairs...")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:  # High correlation threshold
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_val
                })
    
    print(f"Found {len(high_corr_pairs)} feature pairs with |correlation| > 0.8")
    
    if high_corr_pairs:
        print("\n📋 Highly correlated feature pairs:")
        for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:10]:
            print(f"  {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
    
    # Find feature clusters using hierarchical clustering
    print("\n🌳 Finding feature clusters using hierarchical clustering...")
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import squareform
    
    # Convert correlation to distance matrix
    distance_matrix = 1 - np.abs(corr_matrix.values)
    np.fill_diagonal(distance_matrix, 0)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(squareform(distance_matrix), method='ward')
    
    # Find clusters at different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    cluster_results = {}
    
    for threshold in thresholds:
        clusters = fcluster(linkage_matrix, threshold, criterion='distance')
        n_clusters = len(set(clusters))
        cluster_results[threshold] = {
            'n_clusters': n_clusters,
            'clusters': clusters
        }
        print(f"  Threshold {threshold}: {n_clusters} clusters")
    
    # Analyze the most interesting clustering (moderate number of clusters)
    best_threshold = 0.3  # Adjust based on results
    clusters = cluster_results[best_threshold]['clusters']
    n_clusters = cluster_results[best_threshold]['n_clusters']
    
    print(f"\n📊 Analyzing clusters at threshold {best_threshold} ({n_clusters} clusters):")
    
    # Group features by cluster
    feature_clusters = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in feature_clusters:
            feature_clusters[cluster_id] = []
        feature_clusters[cluster_id].append(X.columns[i])
    
    # Print cluster information
    for cluster_id, features in feature_clusters.items():
        print(f"\n  Cluster {cluster_id} ({len(features)} features):")
        if len(features) <= 10:
            for feature in features:
                print(f"    - {feature}")
        else:
            print(f"    - {features[0]} (and {len(features)-1} more)")
    
    # Create correlation heatmap for visualization
    print("\n📈 Creating correlation heatmap...")
    plt.figure(figsize=(12, 10))
    
    # For large datasets, sample features or use a subset
    if X.shape[1] > 50:
        # Sample features for visualization
        sample_features = np.random.choice(X.columns, 50, replace=False)
        corr_subset = corr_matrix.loc[sample_features, sample_features]
        sns.heatmap(corr_subset, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        plt.title(f'Feature Correlation Heatmap (50 random features from {X.shape[1]} total)')
    else:
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Heatmap')
    
    plt.tight_layout()
    plt.savefig('feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("  Saved correlation heatmap to 'feature_correlation_heatmap.png'")
    
    # Create dendrogram
    print("\n🌳 Creating feature dendrogram...")
    plt.figure(figsize=(15, 8))
    
    # For large datasets, use a subset for dendrogram
    if X.shape[1] > 100:
        # Sample features for dendrogram
        sample_indices = np.random.choice(range(len(X.columns)), 100, replace=False)
        sample_features = X.columns[sample_indices]
        sample_corr = corr_matrix.loc[sample_features, sample_features]
        sample_distance = 1 - np.abs(sample_corr.values)
        np.fill_diagonal(sample_distance, 0)
        sample_linkage = linkage(squareform(sample_distance), method='ward')
        
        dendrogram(sample_linkage, labels=sample_features, leaf_rotation=90)
        plt.title(f'Feature Dendrogram (100 random features from {X.shape[1]} total)')
    else:
        dendrogram(linkage_matrix, labels=X.columns, leaf_rotation=90)
        plt.title('Feature Dendrogram')
    
    plt.tight_layout()
    plt.savefig('feature_dendrogram.png', dpi=300, bbox_inches='tight')
    print("  Saved feature dendrogram to 'feature_dendrogram.png'")
    
    # Summary statistics
    print("\n📊 Correlation Summary Statistics:")
    print(f"  Mean absolute correlation: {np.abs(corr_matrix.values).mean():.3f}")
    print(f"  Max absolute correlation: {np.abs(corr_matrix.values).max():.3f}")
    print(f"  Features with |correlation| > 0.5: {(np.abs(corr_matrix.values) > 0.5).sum()}")
    print(f"  Features with |correlation| > 0.8: {(np.abs(corr_matrix.values) > 0.8).sum()}")
    
    # Check for potential multicollinearity
    print("\n⚠️  Multicollinearity Analysis:")
    high_corr_count = (np.abs(corr_matrix.values) > 0.9).sum()
    if high_corr_count > 0:
        print(f"  WARNING: Found {high_corr_count} feature pairs with |correlation| > 0.9")
        print("  Consider removing one feature from each highly correlated pair")
    else:
        print("  No severe multicollinearity detected (|correlation| > 0.9)")
    
    return {
        'correlation_matrix': corr_matrix,
        'high_corr_pairs': high_corr_pairs,
        'feature_clusters': feature_clusters,
        'n_clusters': n_clusters
    }

if __name__ == "__main__":
    results = check_feature_clusters() 