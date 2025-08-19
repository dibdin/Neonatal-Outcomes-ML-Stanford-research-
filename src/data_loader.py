"""
Data loading and preprocessing utilities for gestational age prediction.

This module provides functions for:
- Loading and processing different data types (clinical, biomarker, combined)
- Feature engineering and interaction creation
- Data splitting and preprocessing
- Missing value handling and standardization

Author: Diba Dindoust
Date: 07/01/2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from itertools import combinations
import os
import tempfile
from joblib import Parallel, delayed
import math


def create_interaction_chunk(chunk_data, feature_names, chunk_id, output_dir):
    """
    Create pairwise interactions for a chunk of features.
    
    Args:
        chunk_data (pd.DataFrame): DataFrame containing the chunk of features
        feature_names (list): List of feature names in this chunk
        chunk_id (int): Unique identifier for this chunk
        output_dir (str): Directory to save the chunk file
        
    Returns:
        str: Path to the saved chunk file
    """
    interactions = []
    interaction_names = []
    
    # Create all pairwise interactions within this chunk
    for i, col1 in enumerate(feature_names):
        for j, col2 in enumerate(feature_names[i+1:], i+1):
            interaction_name = f"{col1}_x_{col2}"
            interaction_data = chunk_data[col1] * chunk_data[col2]
            interactions.append(interaction_data)
            interaction_names.append(interaction_name)
    
    # Create DataFrame with interactions
    if interactions:
        interaction_df = pd.DataFrame(dict(zip(interaction_names, interactions)))
        
        # Save to Parquet file
        chunk_file = os.path.join(output_dir, f"interactions_chunk_{chunk_id:04d}.parquet")
        interaction_df.to_parquet(chunk_file, index=False)
        
        print(f"   ✅ Chunk {chunk_id}: Created {len(interactions)} interactions, saved to {chunk_file}")
        return chunk_file
    else:
        print(f"   ⚠️  Chunk {chunk_id}: No interactions created")
        return None


def create_cross_chunk_interactions(chunk1_data, chunk1_names, chunk2_data, chunk2_names, 
                                  chunk_id1, chunk_id2, output_dir):
    """
    Create pairwise interactions between two different chunks of features.
    
    Args:
        chunk1_data (pd.DataFrame): First chunk of features
        chunk1_names (list): Names of features in first chunk
        chunk2_data (pd.DataFrame): Second chunk of features  
        chunk2_names (list): Names of features in second chunk
        chunk_id1 (int): ID of first chunk
        chunk_id2 (int): ID of second chunk
        output_dir (str): Directory to save the chunk file
        
    Returns:
        str: Path to the saved chunk file
    """
    interactions = []
    interaction_names = []
    
    # Create all pairwise interactions between the two chunks
    for col1 in chunk1_names:
        for col2 in chunk2_names:
            interaction_name = f"{col1}_x_{col2}"
            interaction_data = chunk1_data[col1] * chunk2_data[col2]
            interactions.append(interaction_data)
            interaction_names.append(interaction_name)
    
    # Create DataFrame with interactions
    if interactions:
        interaction_df = pd.DataFrame(dict(zip(interaction_names, interactions)))
        
        # Save to Parquet file
        chunk_file = os.path.join(output_dir, f"interactions_cross_{chunk_id1:04d}_{chunk_id2:04d}.parquet")
        interaction_df.to_parquet(chunk_file, index=False)
        
        print(f"   ✅ Cross-chunk {chunk_id1}-{chunk_id2}: Created {len(interactions)} interactions, saved to {chunk_file}")
        return chunk_file
    else:
        print(f"   ⚠️  Cross-chunk {chunk_id1}-{chunk_id2}: No interactions created")
        return None


def create_parallel_pairwise_interactions(data_df, feature_cols, chunk_size=300, n_jobs=-1, 
                                        output_dir=None, recombine=True):
    """
    Create pairwise interactions in parallel with memory-safe chunking.
    
    Args:
        data_df (pd.DataFrame): Input DataFrame with features
        feature_cols (list): List of feature column names
        chunk_size (int): Number of features per chunk (default: 300)
        n_jobs (int): Number of parallel jobs (-1 for all cores)
        output_dir (str): Directory to save chunk files (None for temp directory)
        recombine (bool): Whether to recombine all chunks into single DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with original features + interactions (if recombine=True)
        or list of chunk file paths (if recombine=False)
    """
    print(f"🔍 PARALLEL INTERACTION GENERATION: {len(feature_cols)} features, chunk_size={chunk_size}")
    
    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="interactions_")
    os.makedirs(output_dir, exist_ok=True)
    
    # Split features into chunks
    n_features = len(feature_cols)
    n_chunks = math.ceil(n_features / chunk_size)
    chunks = []
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_features)
        chunk_features = feature_cols[start_idx:end_idx]
        chunk_data = data_df[chunk_features].copy()
        chunks.append((chunk_data, chunk_features, i))
    
    print(f"   📦 Split into {n_chunks} chunks of ~{chunk_size} features each")
    
    # Process chunks in parallel
    chunk_files = []
    
    # 1. Create interactions within each chunk
    print(f"   🔄 Creating interactions within chunks...")
    within_chunk_files = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(create_interaction_chunk)(chunk_data, chunk_features, chunk_id, output_dir)
        for chunk_data, chunk_features, chunk_id in chunks
    )
    chunk_files.extend([f for f in within_chunk_files if f is not None])
    
    # 2. Create interactions between different chunks
    print(f"   🔄 Creating cross-chunk interactions...")
    cross_chunk_files = []
    
    for i in range(n_chunks):
        for j in range(i + 1, n_chunks):  # Only process each pair once
            chunk1_data, chunk1_features, chunk1_id = chunks[i]
            chunk2_data, chunk2_features, chunk2_id = chunks[j]
            
            cross_file = create_cross_chunk_interactions(
                chunk1_data, chunk1_features, 
                chunk2_data, chunk2_features,
                chunk1_id, chunk2_id, output_dir
            )
            if cross_file:
                cross_chunk_files.append(cross_file)
    
    chunk_files.extend(cross_chunk_files)
    
    # Calculate total interactions
    total_within = sum(len(list(combinations(chunk_features, 2))) for _, chunk_features, _ in chunks)
    total_cross = sum(len(chunk1_features) * len(chunk2_features) 
                     for i in range(n_chunks) 
                     for j in range(i + 1, n_chunks)
                     for _, chunk1_features, _ in [chunks[i]]
                     for _, chunk2_features, _ in [chunks[j]])
    total_interactions = total_within + total_cross
    
    print(f"   ✅ Created {total_interactions} total interactions across {len(chunk_files)} files")
    
    if recombine:
        print(f"   🔄 Recombining all chunks into single DataFrame...")
        # Read and combine all chunk files
        interaction_dfs = []
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                df = pd.read_parquet(chunk_file)
                interaction_dfs.append(df)
        
        if interaction_dfs:
            # Combine all interaction DataFrames
            combined_interactions = pd.concat(interaction_dfs, axis=1)
            
            # Combine with original features
            result_df = pd.concat([data_df[feature_cols], combined_interactions], axis=1)
            
            print(f"   ✅ Final DataFrame: {len(feature_cols)} original + {len(combined_interactions.columns)} interaction features")
            
            # Clean up chunk files
            for chunk_file in chunk_files:
                try:
                    os.remove(chunk_file)
                except:
                    pass
            
            return result_df
        else:
            print(f"   ⚠️  No interaction files found, returning original features")
            return data_df[feature_cols]
    else:
        print(f"   📁 Chunk files saved to: {output_dir}")
        return chunk_files


def load_and_process_data(dataset_type='cord', model_type='biomarker', data_option=1, dropna=False, random_state=48, include_all_biomarkers=True, target_type='gestational_age', return_dataframe=False):
    """
    Load and process data for different model types with three data loading options.
    
    Args:
        dataset_type (str): Dataset type ('cord' or 'heel') - only used for naming consistency
        model_type (str): Model type ('clinical', 'biomarker', or 'combined')
        data_option (int): Data loading option:
            1: Load data from Bangladeshcombineddataset_both_samples.csv for patients with both cord and heel data
            2: Load data from BangladeshcombineddatasetJan252022.csv for all heel data entries
            3: Load data from BangladeshcombineddatasetJan252022.csv for all cord data entries
        dropna (bool): Whether to drop rows with missing data
        random_state (int): Random state for reproducibility
        include_all_biomarkers (bool): Whether to include all biomarkers (currently unused)
        target_type (str): Target variable type ('gestational_age' or 'birth_weight')
            - 'gestational_age': Predict gestational age (original pipeline)
            - 'birth_weight': Predict birth weight (SGA classification pipeline)
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target variable
        
    Raises:
        ValueError: If dataset_type, model_type, data_option, or target_type is invalid
    """
    if dataset_type not in ['cord', 'heel']:
        raise ValueError("dataset_type must be either 'cord' or 'heel'.")
    
    if model_type not in ['clinical', 'biomarker', 'combined']:
        raise ValueError("model_type must be either 'clinical', 'biomarker', or 'combined'.")
    
    if data_option not in [1, 2, 3]:
        raise ValueError("data_option must be 1, 2, or 3.")
    
    if target_type not in ['gestational_age', 'birth_weight']:
        raise ValueError("target_type must be either 'gestational_age' or 'birth_weight'.")

    # Load data based on option
    if data_option == 1:
        # Option 1: Load data for patients with both cord and heel data
        df = pd.read_csv('data/Bangladeshcombineddataset_both_samples.csv')
        if 'date_transfusion' in df.columns:
            df = df.drop(columns=['date_transfusion'])
        
        # Split by type
        cord_df = df[df['Source'] == 'CORD'].copy()
        heel_df = df[df['Source'] == 'HEEL'].copy()
        
        # Use the specified dataset type
        if dataset_type == 'cord':
            data_df = cord_df
        else:  # 'heel'
            data_df = heel_df
            
    elif data_option == 2:
        # Option 2: Load all heel data from the full dataset
        df = pd.read_csv('data/BangladeshcombineddatasetJan252022.csv')
        if 'date_transfusion' in df.columns:
            df = df.drop(columns=['date_transfusion'])
        
        # Filter for heel data only
        data_df = df[df['Source'] == 'HEEL'].copy()
        
    elif data_option == 3:
        # Option 3: Load all cord data from the full dataset
        df = pd.read_csv('data/BangladeshcombineddatasetJan252022.csv')
        if 'date_transfusion' in df.columns:
            df = df.drop(columns=['date_transfusion'])
        
        # Filter for cord data only
        data_df = df[df['Source'] == 'CORD'].copy()

    # Drop rows with missing data if specified
    if dropna:
        data_df = data_df.dropna(axis=0, how='any')

    # --- FILTER OUT INVALID BIRTH WEIGHT ROWS ---
    if 'birth_weight_kg' in data_df.columns:
        # Remove rows with birth_weight_kg == 99.9 or <0.5 or >6 kg
        invalid_bw = (data_df['birth_weight_kg'] == 99.9) | (data_df['birth_weight_kg'] < 0.5) | (data_df['birth_weight_kg'] > 6)
        if invalid_bw.any():
            print(f"🔍 PREPROCESSING STEP 1: Filtering out {invalid_bw.sum()} rows with invalid birth_weight_kg values.")
            print(f"   Invalid values found: {data_df.loc[invalid_bw, 'birth_weight_kg'].unique()}")
            data_df = data_df[~invalid_bw]
        else:
            print("✅ PREPROCESSING STEP 1: No invalid birth_weight_kg values found.")

    # --- ENSURE REQUIRED COLUMNS ARE PRESENT ---
    # For gestational age models, ensure birth_weight_kg is in features
    # For birth weight models, ensure gestational_age_weeks is in features
    if model_type in ['clinical', 'combined']:
        if target_type == 'gestational_age' and 'birth_weight_kg' not in data_df.columns:
            raise ValueError('birth_weight_kg must be present in the data for gestational age models.')
        if target_type == 'birth_weight' and 'gestational_age_weeks' not in data_df.columns:
            raise ValueError('gestational_age_weeks must be present in the data for birth weight models.')
        print(f"✅ PREPROCESSING STEP 2: Required columns verified for {target_type} prediction.")

    # Define feature columns based on model type
    if model_type == 'clinical':
        # Clinical/demographic features - adjust based on target_type
        if target_type == 'gestational_age':
            # Original: birth_weight, sex, multiple_birth (columns 145, 146, 158)
            # After dropping date_transfusion (col 157), multiple_birth shifts from 158 to 157
            clinical_cols = [df.columns[144], df.columns[145], df.columns[157]]  # birth_weight, sex, multiple_birth
        else:  # target_type == 'birth_weight'
            # SGA pipeline: gestational_age_weeks, sex, multiple_birth (replace birth_weight with gestational_age_weeks)
            # After dropping date_transfusion (col 157), multiple_birth shifts from 158 to 157
            clinical_cols = ['gestational_age_weeks', df.columns[145], df.columns[157]]  # gestational_age_weeks, sex, multiple_birth
        
        print(f"🔍 PREPROCESSING STEP 3: Base clinical columns: {clinical_cols}")
        
        # Extract base clinical features
        X_clinical = data_df[clinical_cols].copy()
        
        # Create pairwise interactions using parallel processing
        # print(f"🔍 PREPROCESSING STEP 4: Creating pairwise interactions for {len(clinical_cols)} clinical features...")
        # X_with_interactions = create_parallel_pairwise_interactions(
        #     data_df=X_clinical,
        #     feature_cols=clinical_cols,
        #     chunk_size=min(300, len(clinical_cols)),  # Use smaller chunks for clinical features
        #     n_jobs=-1,
        #     recombine=True
        # )
        # 
        # feature_cols = list(X_with_interactions.columns)
        # print(f"✅ PREPROCESSING STEP 4: Created {len(feature_cols) - len(clinical_cols)} clinical interactions. Total features: {len(feature_cols)}")
        
        # For now, use only original features without interactions
        X_with_interactions = X_clinical
        feature_cols = clinical_cols
        print(f"✅ PREPROCESSING STEP 4: Using {len(feature_cols)} clinical features (interactions disabled)")
        
    elif model_type == 'biomarker':
        # Biomarker features (columns 30-141)
        biomarker_cols = df.columns[30:141].tolist()
        print(f"🔍 PREPROCESSING STEP 3: Selected {len(biomarker_cols)} biomarker features")
        
        # Create pairwise interactions using parallel processing
        # print(f"🔍 PREPROCESSING STEP 4: Creating pairwise interactions for {len(biomarker_cols)} biomarkers...")
        X_biomarker = data_df[biomarker_cols].copy()
        
        # Calculate number of interactions (n choose 2)
        # n_biomarkers = len(biomarker_cols)
        # n_interactions = (n_biomarkers * (n_biomarkers - 1)) // 2
        # print(f"   Will create {n_interactions} pairwise interactions...")
        # 
        # X_with_interactions = create_parallel_pairwise_interactions(
        #     data_df=X_biomarker,
        #     feature_cols=biomarker_cols,
        #     chunk_size=300,  # Optimal chunk size for biomarker features
        #     n_jobs=-1,
        #     recombine=True
        # )
        # 
        # feature_cols = list(X_with_interactions.columns)
        # print(f"✅ PREPROCESSING STEP 4: Created {len(feature_cols) - len(biomarker_cols)} biomarker interactions. Total features: {len(feature_cols)}")
        
        # For now, use only original features without interactions
        X_with_interactions = X_biomarker
        feature_cols = biomarker_cols
        print(f"✅ PREPROCESSING STEP 4: Using {len(feature_cols)} biomarker features (interactions disabled)")
        
    elif model_type == 'combined':
        # Both clinical and biomarker features
        biomarker_cols = df.columns[30:141].tolist()
        
        # Clinical features with interactions - adjust based on target_type
        if target_type == 'gestational_age':
            # Original: birth_weight, sex, multiple_birth
            # After dropping date_transfusion (col 157), multiple_birth shifts from 158 to 157
            clinical_cols = [df.columns[144], df.columns[145], df.columns[157]]
        else:  # target_type == 'birth_weight'
            # SGA pipeline: gestational_age_weeks, sex, multiple_birth
            # After dropping date_transfusion (col 157), multiple_birth shifts from 158 to 157
            clinical_cols = ['gestational_age_weeks', df.columns[145], df.columns[157]]
        
        print(f"🔍 PREPROCESSING STEP 3: Selected {len(biomarker_cols)} biomarker features and {len(clinical_cols)} clinical features")
        
        # Extract base clinical features
        X_clinical = data_df[clinical_cols].copy()
        
        # Extract base biomarker features
        X_biomarker = data_df[biomarker_cols].copy()
        
        # Combine all features for pairwise interactions
        all_features = biomarker_cols + clinical_cols
        print(f"🔍 PREPROCESSING STEP 4: Creating pairwise interactions for {len(all_features)} total features...")
        
        # Calculate number of interactions (n choose 2)
        n_features = len(all_features)
        n_interactions = (n_features * (n_features - 1)) // 2
        print(f"   Will create {n_interactions} pairwise interactions...")
        
        # Create pairwise interactions using parallel processing
        # print(f"🔍 PREPROCESSING STEP 4: Creating pairwise interactions for {len(all_features)} total features...")
        # 
        # # Combine all features into single DataFrame for parallel processing
        # X_combined = pd.concat([X_biomarker, X_clinical], axis=1)
        # 
        # X_with_interactions = create_parallel_pairwise_interactions(
        #     data_df=X_combined,
        #     feature_cols=all_features,
        #     chunk_size=300,  # Optimal chunk size for combined features
        #     n_jobs=-1,
        #     recombine=True
        # )
        # 
        # feature_cols = list(X_with_interactions.columns)
        # print(f"✅ PREPROCESSING STEP 4: Created {len(feature_cols) - len(all_features)} pairwise interactions. Total features: {len(feature_cols)}")
        
        # For now, use only original features without interactions
        X_combined = pd.concat([X_biomarker, X_clinical], axis=1)
        X_with_interactions = X_combined
        feature_cols = all_features
        print(f"✅ PREPROCESSING STEP 4: Using {len(feature_cols)} combined features (interactions disabled)")

    # Extract features and labels
    if model_type == 'clinical':
        X = X_with_interactions
        y = data_df['birth_weight_kg'] if target_type == 'birth_weight' else data_df['gestational_age_weeks']
    elif model_type == 'biomarker':
        X = X_with_interactions
        y = data_df['birth_weight_kg'] if target_type == 'birth_weight' else data_df['gestational_age_weeks']
    elif model_type == 'combined':
        # For combined, we need to merge biomarker and clinical data
        y = data_df['birth_weight_kg'] if target_type == 'birth_weight' else data_df['gestational_age_weeks']
        
        # Use the parallel processed data with interactions
        X = X_with_interactions

    # Print information about the features being included
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   Data option: {data_option}")
    print(f"   Dataset: {dataset_type}")
    print(f"   Model type: {model_type}")
    print(f"   Target variable: {target_type}")
    print(f"   Initial features: {len(feature_cols)}")
    print(f"   Initial samples: {len(data_df)}")
    print(f"   X shape: {X.shape}, y shape: {y.shape}")
    
    if model_type == 'clinical':
        print(f"   Clinical features: {feature_cols}")
    elif model_type == 'biomarker':
        print(f"   Biomarkers with TREC: {[col for col in feature_cols if 'TREC' in col]}")
        print(f"   Biomarkers with HGB: {[col for col in feature_cols if 'HGB' in col]}")
    elif model_type == 'combined':
        biomarker_count = len(biomarker_cols)
        clinical_count = len(clinical_cols)
        print(f"   Biomarker features: {biomarker_count}")
        print(f"   Clinical features: {clinical_count}")

    # Initialize feature drop tracking
    feature_drops = {
        'high_missing': [],
        'perfect_correlation': []
    }
    
    print(f"\n🔍 PREPROCESSING STEP 5: MISSING VALUE ANALYSIS")
    print(f"   Initial missing values in X: {X.isnull().sum().sum()}")
    print(f"   Features with missing values: {X.isnull().sum()[X.isnull().sum() > 0].count()}")
    
    # Handle missing values - Drop features with >99% missing
    missing_percentages = (X.isnull().sum() / len(X)) * 100
    high_missing_features = missing_percentages[missing_percentages > 99].index
    if len(high_missing_features) > 0:
        print(f"❌ PREPROCESSING STEP 6: Dropping {len(high_missing_features)} features with >99% missing values:")
        for feat in high_missing_features:
            print(f"   - {feat}: {missing_percentages[feat]:.1f}% missing")
        feature_drops['high_missing'] = list(high_missing_features)
        X = X.drop(columns=high_missing_features)
    else:
        print(f"✅ PREPROCESSING STEP 6: No features with >99% missing values found.")
    
    # Drop columns with all missing values
    all_missing_cols = X.columns[X.isnull().all()].tolist()
    if all_missing_cols:
        print(f"❌ PREPROCESSING STEP 7: Dropping {len(all_missing_cols)} columns with all missing values: {all_missing_cols}")
        X = X.drop(columns=all_missing_cols)
    else:
        print(f"✅ PREPROCESSING STEP 7: No columns with all missing values found.")
    
    print(f"   After missing value cleanup - X shape: {X.shape}")
    

    
    # For clinical features, handle categorical variables
    if model_type in ['clinical', 'combined']:
        # Convert categorical variables to numeric
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"🔍 PREPROCESSING STEP 8: Converting {len(categorical_cols)} categorical variables to numeric: {list(categorical_cols)}")
            for col in categorical_cols:
                if col in X.columns:
                    X[col] = pd.Categorical(X[col]).codes
        else:
            print(f"✅ PREPROCESSING STEP 8: No categorical variables found.")
    
    print(f"   Before imputation - X shape: {X.shape}")
    
    # Use KNNImputer instead of IterativeImputer for faster performance
    from sklearn.impute import KNNImputer
    print(f"🔍 PREPROCESSING STEP 9: Applying KNNImputer (n_neighbors=5)")
    imputer = KNNImputer(n_neighbors=5, weights='uniform')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    print(f"✅ PREPROCESSING STEP 9: Imputation completed. Missing values remaining: {X.isnull().sum().sum()}")
    

    
    # Step 1: Remove perfectly correlated feature pairs (|r| = 1.0)
    print(f"\n🔍 PREPROCESSING STEP 10: Checking for perfectly correlated features (|r| = 1.0)")
    abs_corr = X.corr().abs()
    np.fill_diagonal(abs_corr.values, 0)
    perfect_corr = np.where(abs_corr == 1.0)
    perfect_pairs = [(X.columns[i], X.columns[j]) for i, j in zip(*perfect_corr) if i < j]
    
    if perfect_pairs:
        print(f"❌ PREPROCESSING STEP 10: Found {len(perfect_pairs)} perfectly correlated feature pairs:")
        for pair in perfect_pairs:
            print(f"   - {pair[0]} ↔ {pair[1]}: |r| = 1.0")
        
        # Drop one feature from each perfect pair (keep the one with higher variance)
        perfect_features_to_drop = set()
        for feature1, feature2 in perfect_pairs:
            var1 = X[feature1].var()
            var2 = X[feature2].var()
            if var1 >= var2:
                perfect_features_to_drop.add(feature2)
            else:
                perfect_features_to_drop.add(feature1)
        
        if perfect_features_to_drop:
            print(f"   Dropping {len(perfect_features_to_drop)} perfectly correlated features: {list(perfect_features_to_drop)}")
            feature_drops['perfect_correlation'] = list(perfect_features_to_drop)
            X = X.drop(columns=list(perfect_features_to_drop))
    else:
        print(f"✅ PREPROCESSING STEP 10: No perfectly correlated feature pairs found.")
    
    # Print comprehensive feature drop summary
    print(f"\n📊 FEATURE DROP SUMMARY:")
    total_dropped = sum(len(features) for features in feature_drops.values())
    print(f"   Total features dropped: {total_dropped}")
    for drop_type, features in feature_drops.items():
        if features:
            print(f"   - {drop_type}: {len(features)} features")
            for feat in features:
                print(f"     * {feat}")
    
    print(f"\n✅ FINAL DATASET:")
    print(f"   Final X shape: {X.shape}")
    print(f"   Features remaining: {X.shape[1]}")
    print(f"   Features dropped: {total_dropped}")
    print(f"   Retention rate: {X.shape[1]/(X.shape[1]+total_dropped)*100:.1f}%")

    # --- STANDARDIZE FEATURES ---
    print(f"\n🔍 PREPROCESSING STEP 11: Applying StandardScaler")
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    print(f"✅ PREPROCESSING STEP 11: Standardization completed.")
    print(f"   Feature means after scaling: {X.mean().abs().max():.6f} (should be ~0)")
    print(f"   Feature stds after scaling: {X.std().mean():.6f} (should be ~1)")

    # Handle missing values in target variable
    if y.isna().any():
        print(f"⚠️  WARNING: Found {y.isna().sum()} missing values in target variable. Dropping these samples.")
        # Get indices where target is not NaN
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        if return_dataframe:
            data_df = data_df[valid_indices]
        print(f"   After dropping target missing values - X shape: {X.shape}, y shape: {y.shape}")
    else:
        print(f"✅ Target variable has no missing values.")

    print(f"\n🎯 PREPROCESSING COMPLETE!")
    print(f"   Final dataset: X={X.shape}, y={y.shape}")
    print(f"   Ready for model training.")
    print(f"   " + "="*50)

    if return_dataframe:
        return X, y, data_df
    else:
        return X, y


def load_and_process_sga_data(dataset_type='cord', model_type='biomarker', dropna=False, random_state=48):
    """
    Load and process data for SGA (Small for Gestational Age) regression analysis.
    
    This function creates SGA risk scores based on gestational age percentiles,
    since birth weight data is not available in the current dataset.
    
    Args:
        dataset_type (str): Dataset type ('cord' or 'heel')
        model_type (str): Model type ('clinical', 'biomarker', or 'combined')
        dropna (bool): Whether to drop rows with missing data
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X, y_sga) where X is the feature matrix and y_sga is the SGA risk score
        
    Raises:
        ValueError: If dataset_type or model_type is invalid
    """
    if dataset_type not in ['cord', 'heel']:
        raise ValueError("dataset_type must be either 'cord' or 'heel'.")
    
    if model_type not in ['clinical', 'biomarker', 'combined']:
        raise ValueError("model_type must be either 'clinical', 'biomarker', or 'combined'.")

    # Load data
    df = pd.read_csv('data/Bangladeshcombineddataset_both_samples.csv')

    # Drop rows with missing data if specified
    if dropna:
        df = df.dropna(axis=0, how='any')

    # Split by type
    cord_df = df[df['Source'] == 'CORD'].copy()
    heel_df = df[df['Source'] == 'HEEL'].copy()

    # Define feature columns based on model type (same as original function)
    if model_type == 'clinical':
        # Clinical/demographic features (columns 145, 146, 159 and their pairwise interactions)
        clinical_cols = [df.columns[144], df.columns[145], df.columns[158]]  # 0-indexed, so 145->144, 146->145, 159->158
        print(f"Base clinical columns: {clinical_cols}")
        
        # Get the dataset for this type
        if dataset_type == 'cord':
            data_df = cord_df
        else:  # 'heel'
            data_df = heel_df
            
        # Extract base clinical features
        X_clinical = data_df[clinical_cols].copy()
        
        # Create pairwise interactions
        interaction_features = []
        for col1, col2 in combinations(clinical_cols, 2):
            interaction_name = f"{col1}_x_{col2}"
            X_clinical[interaction_name] = X_clinical[col1] * X_clinical[col2]
            interaction_features.append(interaction_name)
            
        print(f"Created {len(interaction_features)} pairwise interactions: {interaction_features}")
        feature_cols = clinical_cols + interaction_features
        
    elif model_type == 'biomarker':
        # Biomarker features (columns 30-141)
        feature_cols = df.columns[30:141].tolist()
        
    elif model_type == 'combined':
        # Both clinical and biomarker features
        biomarker_cols = df.columns[30:141].tolist()
        
        # Clinical features with interactions
        clinical_cols = [df.columns[144], df.columns[145], df.columns[158]]
        
        # Get the dataset for this type
        if dataset_type == 'cord':
            data_df = cord_df
        else:  # 'heel'
            data_df = heel_df
            
        # Extract base clinical features
        X_clinical = data_df[clinical_cols].copy()
        
        # Create pairwise interactions
        interaction_features = []
        for col1, col2 in combinations(clinical_cols, 2):
            interaction_name = f"{col1}_x_{col2}"
            X_clinical[interaction_name] = X_clinical[col1] * X_clinical[col2]
            interaction_features.append(interaction_name)
            
        clinical_with_interactions = clinical_cols + interaction_features
        feature_cols = biomarker_cols + clinical_with_interactions

    # Extract features and gestational age
    if model_type == 'clinical':
        X = X_clinical
        if dataset_type == 'cord':
            ga_weeks = cord_df['gestational_age_weeks']
        else:  # 'heel'
            ga_weeks = heel_df['gestational_age_weeks']
    elif model_type == 'biomarker':
        if dataset_type == 'cord':
            X = cord_df[feature_cols]
            ga_weeks = cord_df['gestational_age_weeks']
        else:  # 'heel'
            X = heel_df[feature_cols]
            ga_weeks = heel_df['gestational_age_weeks']
    elif model_type == 'combined':
        # For combined, we need to merge biomarker and clinical data
        if dataset_type == 'cord':
            X_biomarker = cord_df[biomarker_cols]
            ga_weeks = cord_df['gestational_age_weeks']
        else:  # 'heel'
            X_biomarker = heel_df[biomarker_cols]
            ga_weeks = heel_df['gestational_age_weeks']
        
        # Merge biomarker and clinical data
        X = pd.concat([X_biomarker, X_clinical], axis=1)

    # Create SGA risk score based on gestational age percentiles
    # Using INTERGROWTH-21st inspired approach
    # SGA is typically <10th percentile, but we'll create a continuous risk score
    
    # Calculate gestational age percentiles
    ga_percentiles = ga_weeks.rank(pct=True) * 100
    
    # Create SGA risk score (higher score = higher risk of SGA)
    # Lower gestational age = higher SGA risk
    sga_risk = 100 - ga_percentiles  # Invert so lower GA = higher risk
    
    # Normalize to 0-1 scale
    sga_risk = (sga_risk - sga_risk.min()) / (sga_risk.max() - sga_risk.min())
    
    # Alternative: Create SGA risk based on distance from expected GA
    # Expected GA is around 40 weeks for term pregnancies
    expected_ga = 40
    ga_deviation = abs(ga_weeks - expected_ga)
    sga_risk_deviation = ga_deviation / ga_deviation.max()
    
    # Combine both approaches
    y_sga = (sga_risk + sga_risk_deviation) / 2

    # Print information about the features being included
    print(f"Dataset: {dataset_type}")
    print(f"Model type: {model_type}")
    print(f"Number of features included: {len(feature_cols)}")
    print(f"SGA risk score range: {y_sga.min():.3f} - {y_sga.max():.3f}")
    print(f"Mean SGA risk: {y_sga.mean():.3f}")
    
    if model_type == 'clinical':
        print(f"Clinical features: {feature_cols}")
    elif model_type == 'biomarker':
        print(f"Biomarkers with TREC: {[col for col in feature_cols if 'TREC' in col]}")
        print(f"Biomarkers with HGB: {[col for col in feature_cols if 'HGB' in col]}")
    elif model_type == 'combined':
        biomarker_count = len(biomarker_cols)
        clinical_count = len(clinical_with_interactions)
        print(f"Biomarker features: {biomarker_count}")
        print(f"Clinical features (with interactions): {clinical_count}")

    # Handle missing values
    X = X.dropna(axis=1, how='all')  # Drop columns with all missing values first
    
    # For clinical features, handle categorical variables
    if model_type in ['clinical', 'combined']:
        # Convert categorical variables to numeric
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes
    
    # Impute missing values with column-wise mean
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    return X, y_sga


def split_data(X, y, test_size, random_state=None):
    """
    Split data into training and test sets.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        test_size (float): Proportion of data to use for testing
        random_state (int, optional): Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)