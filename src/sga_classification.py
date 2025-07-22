"""
SGA (Small for Gestational Age) Classification Utilities.

This module provides functions for classifying babies as SGA based on 
Intergrowth-21 chart reference values for birth weight by gestational age and sex.

Author: Diba Dindoust
Date: 07/01/2025
"""

import pandas as pd
import numpy as np
from scipy import stats

# Intergrowth-21 chart reference values for 3rd percentile SGA classification
# Source: PI provided Intergrowth-21 chart values
INTERGROWTH21_3RD_PERCENTILE = {
    'gestational_age_weeks': list(range(24, 47)),  # 24 to 46 weeks
    'boys_birth_weight_g': [470, 540, 610, 690, 790, 890, 1000, 1130, 1270, 1300, 1560, 1800, 2020, 2220, 2390, 2550, 2690, 2820, 2920, 3300, 3500, 3700, 3900],
    'girls_birth_weight_g': [440, 510, 580, 660, 740, 840, 950, 1070, 1200, 1320, 1570, 1800, 2000, 2180, 2340, 2480, 2600, 2690, 2770, 3050, 3200, 3400, 3580]
}

def get_intergrowth21_threshold(gestational_age, sex):
    """
    Get the Intergrowth-21 3rd percentile birth weight threshold for a given gestational age and sex.
    
    Args:
        gestational_age (float): Gestational age in weeks
        sex (int): Sex (1 for male, 2 for female)
    
    Returns:
        float: Birth weight threshold in grams, or None if outside range
    """
    # Round gestational age to nearest week
    ga_rounded = round(gestational_age)
    
    # Check if gestational age is within the Intergrowth-21 range (24-46 weeks)
    if ga_rounded < 24 or ga_rounded > 46:
        return None
    
    # Get index for the gestational age
    ga_index = ga_rounded - 24
    
    # Get the appropriate threshold based on sex
    if sex == 1:  # Male
        return INTERGROWTH21_3RD_PERCENTILE['boys_birth_weight_g'][ga_index]
    elif sex == 2:  # Female
        return INTERGROWTH21_3RD_PERCENTILE['girls_birth_weight_g'][ga_index]
    else:
        # If sex is unknown, use average of male and female thresholds
        male_threshold = INTERGROWTH21_3RD_PERCENTILE['boys_birth_weight_g'][ga_index]
        female_threshold = INTERGROWTH21_3RD_PERCENTILE['girls_birth_weight_g'][ga_index]
        return (male_threshold + female_threshold) / 2

def calculate_sga_classification_intergrowth21(birth_weights, gestational_ages, sexes):
    """
    Calculate SGA classification using Intergrowth-21 3rd percentile reference values.
    
    Args:
        birth_weights (array-like): Birth weights in kg
        gestational_ages (array-like): Gestational ages in weeks
        sexes (array-like): Sex values (1 for male, 2 for female)
    
    Returns:
        array: Binary classification (1 = SGA, 0 = Normal)
    """
    # Convert birth weights from kg to grams
    birth_weights_g = np.array(birth_weights) * 1000
    
    sga_classification = np.zeros(len(birth_weights_g))
    
    for i, (bw_g, ga, sex) in enumerate(zip(birth_weights_g, gestational_ages, sexes)):
        # Get the Intergrowth-21 threshold for this gestational age and sex
        threshold_g = get_intergrowth21_threshold(ga, sex)
        
        if threshold_g is not None:
            # Classify as SGA if birth weight is below the 3rd percentile threshold
            sga_classification[i] = 1 if bw_g < threshold_g else 0
        else:
            # If gestational age is outside Intergrowth-21 range, mark as missing
            sga_classification[i] = np.nan
    
    return sga_classification

def calculate_sga_classification_10th_percentile_intergrowth21(birth_weights, gestational_ages, sexes):
    """
    Calculate SGA classification using estimated 10th percentile from Intergrowth-21.
    Since Intergrowth-21 only provides 3rd percentile, we estimate 10th percentile
    by adding approximately 15% to the 3rd percentile threshold.
    
    Args:
        birth_weights (array-like): Birth weights in kg
        gestational_ages (array-like): Gestational ages in weeks
        sexes (array-like): Sex values (1 for male, 2 for female)
    
    Returns:
        array: Binary classification (1 = SGA, 0 = Normal)
    """
    # Convert birth weights from kg to grams
    birth_weights_g = np.array(birth_weights) * 1000
    
    sga_classification = np.zeros(len(birth_weights_g))
    
    for i, (bw_g, ga, sex) in enumerate(zip(birth_weights_g, gestational_ages, sexes)):
        # Get the Intergrowth-21 3rd percentile threshold
        threshold_3rd_g = get_intergrowth21_threshold(ga, sex)
        
        if threshold_3rd_g is not None:
            # Estimate 10th percentile by adding ~15% to 3rd percentile
            # This is an approximation based on typical birth weight distributions
            threshold_10th_g = threshold_3rd_g * 1.15
            
            # Classify as SGA if birth weight is below the estimated 10th percentile threshold
            sga_classification[i] = 1 if bw_g < threshold_10th_g else 0
        else:
            # If gestational age is outside Intergrowth-21 range, mark as missing
            sga_classification[i] = np.nan
    
    return sga_classification

def get_gestational_age_and_sex_data(data_option, dataset_type):
    """
    Load gestational age and sex data for SGA classification.
    
    Args:
        data_option (int): Data loading option (1, 2, or 3)
        dataset_type (str): Dataset type ('cord' or 'heel')
    
    Returns:
        tuple: (gestational_ages, sexes) arrays
    """
    from src.data_loader import load_and_process_data
    
    # Load data with gestational age as target to get the gestational age values and original dataframe
    X, ga, data_df = load_and_process_data(dataset_type, model_type='clinical', data_option=data_option, target_type='gestational_age', return_dataframe=True)
    
    # Get sex column (column 145, which is 0-indexed as 144)
    sex_col = data_df.columns[144]  # sex column
    sexes = data_df[sex_col].values
    
    return ga.values, sexes

def create_sga_targets_intergrowth21(birth_weights, data_option, dataset_type, sga_type='3rd_percentile'):
    """
    Create SGA classification targets using Intergrowth-21 reference values.
    
    Args:
        birth_weights (array-like): Birth weights in kg
        data_option (int): Data loading option (1, 2, or 3)
        dataset_type (str): Dataset type ('cord' or 'heel')
        sga_type (str): Type of SGA classification ('3rd_percentile' or '10th_percentile')
    
    Returns:
        array: Binary SGA classification (1 = SGA, 0 = Normal)
    """
    # Get gestational ages and sexes for the same samples
    gestational_ages, sexes = get_gestational_age_and_sex_data(data_option, dataset_type)
    
    # Calculate SGA classification using Intergrowth-21
    if sga_type == '10th_percentile':
        return calculate_sga_classification_10th_percentile_intergrowth21(birth_weights, gestational_ages, sexes)
    else:  # 3rd_percentile
        return calculate_sga_classification_intergrowth21(birth_weights, gestational_ages, sexes)

# Legacy functions for backward compatibility (keeping the old percentile-based approach)
def calculate_sga_classification(birth_weights, gestational_ages, percentile_cutoff=10):
    """
    DEPRECATED: Calculate SGA classification based on dataset percentiles.
    Use calculate_sga_classification_intergrowth21() instead for more accurate classification.
    
    Args:
        birth_weights (array-like): Birth weights in kg
        gestational_ages (array-like): Gestational ages in weeks
        percentile_cutoff (int): Percentile cutoff for SGA classification (default: 10 for 10th percentile)
    
    Returns:
        array: Binary classification (1 = SGA, 0 = Normal)
    """
    # Create DataFrame for easier manipulation
    df = pd.DataFrame({
        'birth_weight': birth_weights,
        'gestational_age': gestational_ages
    })
    
    # Calculate percentiles for each gestational age
    sga_classification = np.zeros(len(df))
    
    # Group by gestational age (rounded to nearest week for percentile calculation)
    df['ga_rounded'] = df['gestational_age'].round()
    
    for ga_week in df['ga_rounded'].unique():
        if pd.isna(ga_week):
            continue
            
        # Get data for this gestational age
        ga_mask = df['ga_rounded'] == ga_week
        ga_data = df[ga_mask]['birth_weight']
        
        if len(ga_data) < 5:  # Need at least 5 samples for percentile calculation
            # Use overall percentile if not enough data for this GA
            overall_percentile = np.percentile(df['birth_weight'].dropna(), percentile_cutoff)
            sga_classification[ga_mask] = (ga_data < overall_percentile).astype(int)
        else:
            # Calculate percentile for this specific gestational age
            ga_percentile = np.percentile(ga_data, percentile_cutoff)
            sga_classification[ga_mask] = (ga_data < ga_percentile).astype(int)
    
    return sga_classification

def calculate_sga_classification_3rd_percentile(birth_weights, gestational_ages):
    """
    DEPRECATED: Calculate SGA classification using 3rd percentile cutoff.
    Use calculate_sga_classification_intergrowth21() instead.
    
    Args:
        birth_weights (array-like): Birth weights in kg
        gestational_ages (array-like): Gestational ages in weeks
    
    Returns:
        array: Binary classification (1 = SGA, 0 = Normal)
    """
    return calculate_sga_classification(birth_weights, gestational_ages, percentile_cutoff=3)

def calculate_sga_classification_10th_percentile(birth_weights, gestational_ages):
    """
    DEPRECATED: Calculate SGA classification using 10th percentile cutoff.
    Use calculate_sga_classification_10th_percentile_intergrowth21() instead.
    
    Args:
        birth_weights (array-like): Birth weights in kg
        gestational_ages (array-like): Gestational ages in weeks
    
    Returns:
        array: Binary classification (1 = SGA, 0 = Normal)
    """
    return calculate_sga_classification(birth_weights, gestational_ages, percentile_cutoff=10)

def get_gestational_age_data(data_option, dataset_type):
    """
    DEPRECATED: Load gestational age data for SGA classification.
    Use get_gestational_age_and_sex_data() instead.
    
    Args:
        data_option (int): Data loading option (1, 2, or 3)
        dataset_type (str): Dataset type ('cord' or 'heel')
    
    Returns:
        array: Gestational ages for the corresponding samples
    """
    from src.data_loader import load_and_process_data
    
    # Load data with gestational age as target to get the gestational age values
    X, ga = load_and_process_data(dataset_type, model_type='clinical', data_option=data_option, target_type='gestational_age')
    
    return ga.values

def create_sga_targets(birth_weights, data_option, dataset_type, sga_type='10th_percentile'):
    """
    DEPRECATED: Create SGA classification targets using dataset percentiles.
    Use create_sga_targets_intergrowth21() instead for more accurate classification.
    
    Args:
        birth_weights (array-like): Birth weights in kg
        data_option (int): Data loading option (1, 2, or 3)
        dataset_type (str): Dataset type ('cord' or 'heel')
        sga_type (str): Type of SGA classification ('10th_percentile' or '3rd_percentile')
    
    Returns:
        array: Binary SGA classification (1 = SGA, 0 = Normal)
    """
    # Get gestational ages for the same samples
    gestational_ages = get_gestational_age_data(data_option, dataset_type)
    
    # Calculate SGA classification
    if sga_type == '3rd_percentile':
        return calculate_sga_classification_3rd_percentile(birth_weights, gestational_ages)
    else:  # 10th_percentile
        return calculate_sga_classification_10th_percentile(birth_weights, gestational_ages) 