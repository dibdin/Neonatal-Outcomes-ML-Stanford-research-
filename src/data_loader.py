import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from itertools import combinations

def load_and_process_data(dataset_type='cord', model_type='biomarker', dropna=False, random_state=48, include_all_biomarkers=True):
    """
    Load and process data for different model types:
    - 'clinical': Only clinical/demographic features (columns 145, 146, 159 and their pairwise interactions)
    - 'biomarker': Only biomarker features (current model)
    - 'combined': Both clinical and biomarker features
    """
    if dataset_type not in ['cord', 'heel']:
        raise ValueError("dataset_type must be either 'cord' or 'heel'.")
    
    if model_type not in ['clinical', 'biomarker', 'combined']:
        raise ValueError("model_type must be either 'clinical', 'biomarker', or 'combined'.")

    # Load data
    df = pd.read_csv('data/BangladeshcombineddatasetJan252022.csv')

    # Drop rows with missing data if specified
    if dropna:
        df = df.dropna(axis=0, how='any')

    # Split by type
    cord_df = df[df['Source'] == 'CORD'].copy()
    heel_df = df[df['Source'] == 'HEEL'].copy()

    # Define feature columns based on model type
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

    # Extract features and labels
    if model_type == 'clinical':
        X = X_clinical
        if dataset_type == 'cord':
            y = cord_df['gestational_age_weeks']
        else:  # 'heel'
            y = heel_df['gestational_age_weeks']
    elif model_type == 'biomarker':
        if dataset_type == 'cord':
            X = cord_df[feature_cols]
            y = cord_df['gestational_age_weeks']
        else:  # 'heel'
            X = heel_df[feature_cols]
            y = heel_df['gestational_age_weeks']
    elif model_type == 'combined':
        # For combined, we need to merge biomarker and clinical data
        if dataset_type == 'cord':
            X_biomarker = cord_df[biomarker_cols]
            y = cord_df['gestational_age_weeks']
        else:  # 'heel'
            X_biomarker = heel_df[biomarker_cols]
            y = heel_df['gestational_age_weeks']
        
        # Merge biomarker and clinical data
        X = pd.concat([X_biomarker, X_clinical], axis=1)

    # Print information about the features being included
    print(f"Dataset: {dataset_type}")
    print(f"Model type: {model_type}")
    print(f"Number of features included: {len(feature_cols)}")
    
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

    # Standardize the features (important for regularized models like ElasticNet and STABL)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    return X, y

def split_data(X, y, test_size, random_state=None):  
        return train_test_split(X, y, test_size=test_size, random_state=random_state)