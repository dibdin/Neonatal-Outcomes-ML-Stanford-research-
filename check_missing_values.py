#!/usr/bin/env python3
"""
Check missing values and imputation statistics for every dataset and model
"""

import pandas as pd
import numpy as np
from src.data_loader import load_and_process_data
import warnings
warnings.filterwarnings('ignore')

def check_missing_values():
    """Check missing values for all datasets and models"""
    
    print("=== MISSING VALUES ANALYSIS FOR ALL DATASETS AND MODELS ===\n")
    
    # Define all combinations to test
    datasets = ['cord', 'heel']
    model_types = ['clinical', 'biomarker', 'combined']
    data_options = [1, 2, 3]
    target_types = ['gestational_age', 'birth_weight']
    
    results = []
    
    for dataset in datasets:
        for model_type in model_types:
            for data_option in data_options:
                for target_type in target_types:
                    try:
                        print(f"🔍 Checking: {dataset.upper()} | {model_type.upper()} | Data Option {data_option} | {target_type.upper()}")
                        print("-" * 80)
                        
                        # Load data with return_dataframe=True to get raw data
                        X, y, data_df = load_and_process_data(
                            dataset_type=dataset,
                            model_type=model_type,
                            data_option=data_option,
                            dropna=False,  # Don't drop missing values initially
                            random_state=48,
                            target_type=target_type,
                            return_dataframe=True
                        )
                        
                        # Get feature columns based on model type
                        if model_type == 'clinical':
                            # Clinical features with interactions
                            if target_type == 'gestational_age':
                                clinical_cols = [data_df.columns[144], data_df.columns[145], data_df.columns[158]]
                            else:  # birth_weight
                                clinical_cols = ['gestational_age_weeks', data_df.columns[145], data_df.columns[158]]
                            
                            # Create interaction features
                            from itertools import combinations
                            interaction_features = []
                            for col1, col2 in combinations(clinical_cols, 2):
                                interaction_name = f"{col1}_x_{col2}"
                                interaction_features.append(interaction_name)
                            
                            feature_cols = clinical_cols + interaction_features
                            
                        elif model_type == 'biomarker':
                            # Biomarker features (columns 30-141)
                            feature_cols = data_df.columns[30:141].tolist()
                            
                        elif model_type == 'combined':
                            # Both biomarker and clinical features
                            biomarker_cols = data_df.columns[30:141].tolist()
                            
                            if target_type == 'gestational_age':
                                clinical_cols = [data_df.columns[144], data_df.columns[145], data_df.columns[158]]
                            else:  # birth_weight
                                clinical_cols = ['gestational_age_weeks', data_df.columns[145], data_df.columns[158]]
                            
                            # Create interaction features
                            from itertools import combinations
                            interaction_features = []
                            for col1, col2 in combinations(clinical_cols, 2):
                                interaction_name = f"{col1}_x_{col2}"
                                interaction_features.append(interaction_name)
                            
                            clinical_with_interactions = clinical_cols + interaction_features
                            feature_cols = biomarker_cols + clinical_with_interactions
                        
                        # Extract features from raw data
                        X_raw = data_df[feature_cols].copy()
                        
                        # Check missing values in raw data
                        missing_counts = X_raw.isnull().sum()
                        missing_percentages = (missing_counts / len(X_raw)) * 100
                        
                        # Count columns with missing values
                        columns_with_missing = missing_counts[missing_counts > 0]
                        total_missing_columns = len(columns_with_missing)
                        
                        # Count total missing values
                        total_missing_values = missing_counts.sum()
                        
                        # Check target variable missing values
                        target_missing = y.isnull().sum()
                        target_missing_pct = (target_missing / len(y)) * 100
                        
                        # Print results
                        print(f"📊 Dataset Statistics:")
                        print(f"   Total samples: {len(data_df)}")
                        print(f"   Total features: {len(feature_cols)}")
                        print(f"   Features with missing values: {total_missing_columns}")
                        print(f"   Total missing values: {total_missing_values}")
                        print(f"   Missing values percentage: {(total_missing_values / (len(X_raw) * len(X_raw.columns))) * 100:.2f}%")
                        print(f"   Target missing values: {target_missing} ({target_missing_pct:.2f}%)")
                        
                        if total_missing_columns > 0:
                            print(f"\n📋 Features with missing values:")
                            for col in columns_with_missing.index[:10]:  # Show first 10
                                missing_count = missing_counts[col]
                                missing_pct = missing_percentages[col]
                                print(f"   {col}: {missing_count} missing ({missing_pct:.1f}%)")
                            
                            if len(columns_with_missing) > 10:
                                print(f"   ... and {len(columns_with_missing) - 10} more features")
                        
                        # Store results
                        results.append({
                            'dataset': dataset,
                            'model_type': model_type,
                            'data_option': data_option,
                            'target_type': target_type,
                            'total_samples': len(data_df),
                            'total_features': len(feature_cols),
                            'features_with_missing': total_missing_columns,
                            'total_missing_values': total_missing_values,
                            'missing_percentage': (total_missing_values / (len(X_raw) * len(X_raw.columns))) * 100,
                            'target_missing': target_missing,
                            'target_missing_pct': target_missing_pct
                        })
                        
                        print(f"✅ Analysis complete\n")
                        
                    except Exception as e:
                        print(f"❌ Error analyzing {dataset} | {model_type} | Data Option {data_option} | {target_type}: {str(e)}")
                        print()
                        continue
    
    # Create summary DataFrame
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("📊 SUMMARY STATISTICS")
    print("="*80)
    
    if not results_df.empty:
        print(f"Total configurations analyzed: {len(results_df)}")
        print(f"Average missing percentage across all datasets: {results_df['missing_percentage'].mean():.2f}%")
        print(f"Average target missing percentage: {results_df['target_missing_pct'].mean():.2f}%")
        
        print(f"\n📈 Missing Values by Model Type:")
        for model_type in results_df['model_type'].unique():
            subset = results_df[results_df['model_type'] == model_type]
            print(f"   {model_type.upper()}: {subset['missing_percentage'].mean():.2f}% average missing")
        
        print(f"\n📈 Missing Values by Dataset:")
        for dataset in results_df['dataset'].unique():
            subset = results_df[results_df['dataset'] == dataset]
            print(f"   {dataset.upper()}: {subset['missing_percentage'].mean():.2f}% average missing")
        
        print(f"\n📈 Missing Values by Target Type:")
        for target_type in results_df['target_type'].unique():
            subset = results_df[results_df['target_type'] == target_type]
            print(f"   {target_type.upper()}: {subset['missing_percentage'].mean():.2f}% average missing")
        
        # Show configurations with highest missing values
        print(f"\n⚠️  Configurations with highest missing values:")
        high_missing = results_df.nlargest(5, 'missing_percentage')
        for _, row in high_missing.iterrows():
            print(f"   {row['dataset']} | {row['model_type']} | Data Option {row['data_option']} | {row['target_type']}: {row['missing_percentage']:.2f}% missing")
        
        # Show configurations with no missing values
        print(f"\n✅ Configurations with no missing values:")
        no_missing = results_df[results_df['missing_percentage'] == 0]
        if not no_missing.empty:
            for _, row in no_missing.iterrows():
                print(f"   {row['dataset']} | {row['model_type']} | Data Option {row['data_option']} | {row['target_type']}")
        else:
            print("   None found")
    
    return results_df

if __name__ == "__main__":
    results = check_missing_values() 