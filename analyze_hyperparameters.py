import pickle
import numpy as np
from collections import defaultdict

def analyze_hyperparameters():
    """Analyze hyperparameter frequencies from the most recent run."""
    
    print("=== HYPERPARAMETER FREQUENCY ANALYSIS ===")
    print("\nLoading data from all_results.pkl...")
    
    # Load the results
    with open('all_results.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Initialize counters
    hyperparams = {
        'alpha': defaultdict(int),
        'l1_ratio': defaultdict(int),
        'C': defaultdict(int)
    }
    
    model_hyperparams = {}
    
    print("\nExtracting hyperparameters from all models...")
    
    # Extract hyperparameters from all gestational_age models
    for key, result in data.items():
        if 'gestational_age' in key:
            print(f"\nAnalyzing: {key}")
            
            # Get predictions from this model
            predictions = result.get('predictions', [])
            
            for i, pred in enumerate(predictions):
                print(f"  Run {i+1}:")
                
                # Extract alpha values
                if 'optimized_alpha' in pred and pred['optimized_alpha'] is not None:
                    alpha = pred['optimized_alpha']
                    hyperparams['alpha'][alpha] += 1
                    print(f"    Alpha: {alpha}")
                
                # Extract l1_ratio values
                if 'optimized_l1_ratio' in pred and pred['optimized_l1_ratio'] is not None:
                    l1_ratio = pred['optimized_l1_ratio']
                    hyperparams['l1_ratio'][l1_ratio] += 1
                    print(f"    L1_ratio: {l1_ratio}")
                
                # Store model-specific hyperparameters
                if key not in model_hyperparams:
                    model_hyperparams[key] = []
                model_hyperparams[key].append({
                    'alpha': pred.get('optimized_alpha'),
                    'l1_ratio': pred.get('optimized_l1_ratio')
                })
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("HYPERPARAMETER FREQUENCY SUMMARY")
    print(f"{'='*60}")
    
    for param_type, values in hyperparams.items():
        if values:
            print(f"\n{param_type.upper()}:")
            sorted_values = sorted(values.items(), key=lambda x: x[0])
            total = sum(values.values())
            
            for value, count in sorted_values:
                percentage = (count / total) * 100
                print(f"  {value}: {count} times ({percentage:.1f}%)")
            
            print(f"  Total: {total} observations")
    
    # Print model-specific analysis
    print(f"\n{'='*60}")
    print("MODEL-SPECIFIC HYPERPARAMETER ANALYSIS")
    print(f"{'='*60}")
    
    for model_key, runs in model_hyperparams.items():
        print(f"\n{model_key}:")
        for i, run in enumerate(runs):
            print(f"  Run {i+1}: Alpha={run['alpha']}, L1_ratio={run['l1_ratio']}")
    
    # Calculate statistics
    print(f"\n{'='*60}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*60}")
    
    if hyperparams['alpha']:
        alphas = list(hyperparams['alpha'].keys())
        print(f"\nAlpha Statistics:")
        print(f"  Range: {min(alphas)} - {max(alphas)}")
        print(f"  Mean: {np.mean(alphas):.4f}")
        print(f"  Median: {np.median(alphas):.4f}")
        print(f"  Most common: {max(hyperparams['alpha'].items(), key=lambda x: x[1])}")
    
    if hyperparams['l1_ratio']:
        l1_ratios = list(hyperparams['l1_ratio'].keys())
        print(f"\nL1_ratio Statistics:")
        print(f"  Range: {min(l1_ratios)} - {max(l1_ratios)}")
        print(f"  Mean: {np.mean(l1_ratios):.4f}")
        print(f"  Median: {np.median(l1_ratios):.4f}")
        print(f"  Most common: {max(hyperparams['l1_ratio'].items(), key=lambda x: x[1])}")

if __name__ == "__main__":
    analyze_hyperparameters() 