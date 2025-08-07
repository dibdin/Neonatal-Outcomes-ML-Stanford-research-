#!/bin/bash
#SBATCH --job-name=serum_biomarkers_complete
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal

# Load Python module
module load python/3.9

# Install required packages if needed
pip install --user pandas numpy scikit-learn matplotlib seaborn adjustText

echo "🚀 Starting COMPREHENSIVE serum biomarkers analysis..."
echo "📊 Using configuration from src/config.py"
echo "🔧 N_REPEATS: 100, TEST_SIZE: 0.2, PRETERM_CUTOFF: 37"

# Step 1: Run gestational age analysis (using config values)
echo "📊 Step 1: Running gestational age analysis..."
python main.py gestational_age
echo "✅ Gestational age analysis completed"

# Step 2: Run birth weight analysis (using config values)
echo "📊 Step 2: Running birth weight analysis..."
python main.py birth_weight
echo "✅ Birth weight analysis completed"

# Step 3: Merge all results
echo "📊 Step 3: Merging all results..."
python merge_all_results.py
echo "✅ Results merged"

# Step 4: Generate comparison plots
echo "📊 Step 4: Generating comparison plots..."
python generate_comparison_plots.py
echo "✅ Comparison plots generated"

# Step 5: Organize plots
echo "📊 Step 5: Organizing plots..."
python organize_plots.py
echo "✅ Plots organized"

# Step 6: Run additional analysis scripts
echo "📊 Step 6: Running additional analyses..."

# Analyze hyperparameters
echo "  - Analyzing hyperparameters..."
python analyze_hyperparameters.py

# Generate comprehensive summaries
echo "  - Generating comprehensive summaries..."
python comprehensive_hyperparameter_summary.py

# Run pipeline summary
echo "  - Running pipeline summary..."
python pipeline_summary.py

# Generate plot summary
echo "  - Generating plot summary..."
python plot_summary.py

echo "🎉 ALL ANALYSES COMPLETED at $(date)"
echo "📊 Configuration used:"
echo "  - N_REPEATS: 100"
echo "  - TEST_SIZE: 0.2" 
echo "  - PRETERM_CUTOFF: 37"
