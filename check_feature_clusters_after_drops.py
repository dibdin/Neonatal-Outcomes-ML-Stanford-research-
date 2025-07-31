#!/usr/bin/env python3
"""
Re-evaluate feature clusters after implementing feature drops
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_and_process_data
import warnings
warnings.filterwarnings('ignore')

def check_feature_clusters_after_drops():
    """Check feature clusters after implementing feature drops"""
    
    print("=== FEATURE CLUSTER ANALYSIS AFTER DROPS ===\n")
    
    # Test with biomarker data (most features)
    print("🔍 Loading biomarker data with feature drops...")
    
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
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Number of features after drops: {X.shape[1]}")
    
    # Calculate correlation matrix
    print("\n📊 Calculating correlation matrix...")
    corr_matrix = X.corr()
    
    # Find highly correlated feature pairs
    print("\n🔍 Finding highly correlated feature pairs...")
    high_corr_pairs = []
    
    # Use numpy operations for speed
    corr_values = corr_matrix.values
    feature_names = corr_matrix.columns
    
    # Find upper triangle indices where correlation > 0.8
    upper_triangle = np.triu(np.ones_like(corr_values), k=1).astype(bool)
    high_corr_mask = (np.abs(corr_values) > 0.8) & upper_triangle
    
    high_corr_indices = np.where(high_corr_mask)
    
    for i, j in zip(high_corr_indices[0], high_corr_indices[1]):
        high_corr_pairs.append({
            'feature1': feature_names[i],
            'feature2': feature_names[j],
            'correlation': corr_values[i, j]
        })
    
    print(f"Found {len(high_corr_pairs)} feature pairs with |correlation| > 0.8")
    
    if high_corr_pairs:
        print("\n📋 Top 10 highly correlated feature pairs:")
        for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:10]:
            print(f"  {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
    
    # Fast clustering using correlation-based distance
    print("\n🌳 Finding feature clusters using correlation-based clustering...")
    
    # Use a simpler clustering approach for speed
    from sklearn.cluster import AgglomerativeClustering
    
    # Convert correlation to distance matrix
    distance_matrix = 1 - np.abs(corr_values)
    np.fill_diagonal(distance_matrix, 0)
    
    # Handle NaN values in distance matrix
    distance_matrix = np.nan_to_num(distance_matrix, nan=1.0)  # Replace NaN with max distance
    
    # Use AgglomerativeClustering which is faster than scipy linkage
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=0.3,  # Cluster features with distance < 0.3
        linkage='ward',
        compute_distances=True
    )
    
    clusters = clustering.fit_predict(distance_matrix)
    n_clusters = len(set(clusters))
    
    print(f"Found {n_clusters} feature clusters")
    
    # Group features by cluster
    feature_clusters = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in feature_clusters:
            feature_clusters[cluster_id] = []
        feature_clusters[cluster_id].append(feature_names[i])
    
    # Print cluster information
    print(f"\n📊 Feature clusters after drops:")
    for cluster_id, features in feature_clusters.items():
        print(f"\n  Cluster {cluster_id} ({len(features)} features):")
        if len(features) <= 5:
            for feature in features:
                print(f"    - {feature}")
        else:
            print(f"    - {features[0]} (and {len(features)-1} more)")
    
    # Create correlation heatmap for visualization (sampled)
    print("\n📈 Creating correlation heatmap...")
    plt.figure(figsize=(12, 10))
    
    # For visualization, use a smaller subset
    if X.shape[1] > 30:
        viz_features = np.random.choice(X.columns, 30, replace=False)
        corr_viz = corr_matrix.loc[viz_features, viz_features]
        sns.heatmap(corr_viz, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        plt.title(f'Feature Correlation Heatmap After Drops (30 random features from {X.shape[1]} total)')
    else:
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Heatmap After Drops')
    
    plt.tight_layout()
    plt.savefig('feature_correlation_heatmap_after_drops.png', dpi=300, bbox_inches='tight')
    print("  Saved correlation heatmap to 'feature_correlation_heatmap_after_drops.png'")
    
    # Summary statistics
    print("\n📊 Correlation Summary Statistics After Drops:")
    print(f"  Mean absolute correlation: {np.abs(corr_values).mean():.3f}")
    print(f"  Max absolute correlation: {np.abs(corr_values).max():.3f}")
    print(f"  Features with |correlation| > 0.5: {(np.abs(corr_values) > 0.5).sum()}")
    print(f"  Features with |correlation| > 0.8: {(np.abs(corr_values) > 0.8).sum()}")
    
    # Check for potential multicollinearity
    print("\n⚠️  Multicollinearity Analysis After Drops:")
    high_corr_count = (np.abs(corr_values) > 0.9).sum()
    if high_corr_count > 0:
        print(f"  WARNING: Found {high_corr_count} feature pairs with |correlation| > 0.9")
        print("  Consider removing one feature from each highly correlated pair")
    else:
        print("  ✅ No severe multicollinearity detected (|correlation| > 0.9)")
    
    # Additional analysis: feature variance
    print("\n📊 Feature Variance Analysis After Drops:")
    feature_variances = X.var()
    low_var_features = feature_variances[feature_variances < 0.01]
    print(f"  Features with variance < 0.01: {len(low_var_features)}")
    if len(low_var_features) > 0:
        print("  ⚠️  Still have low-variance features")
    else:
        print("  ✅ No low-variance features remaining")
    
    # Compare with original analysis
    print("\n📊 Comparison with Original Analysis:")
    print("  Original features: 105")
    print(f"  Features after drops: {X.shape[1]}")
    print(f"  Features dropped: {105 - X.shape[1]}")
    
    return {
        'correlation_matrix': corr_matrix,
        'high_corr_pairs': high_corr_pairs,
        'feature_clusters': feature_clusters,
        'n_clusters': n_clusters,
        'feature_variances': feature_variances,
        'final_feature_count': X.shape[1]
    }

if __name__ == "__main__":
    results = check_feature_clusters_after_drops() 