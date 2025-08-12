#!/bin/bash

# ----------------------------------------
# 🔧 CONFIGURATION
# ----------------------------------------
USER="dibadin"
HOST="sherlock.stanford.edu"
REMOTE_DIR="/home/users/dibadin/serum_biomarkers"
LOCAL_SCRIPT="main.py"
JOB_SCRIPT="job.slurm"
JOB_NAME="serum_biomarkers_complete"
OUTPUT_FILE="output_%j.txt"
ERROR_FILE="error_%j.txt"
TIME_LIMIT="72:00:00"
MEMORY="64G"
CPUS=16
PARTITION="normal"
PYTHON_MODULE="python/3.9"

# ----------------------------------------
# 📦 STEP 1: Create job script
# ----------------------------------------
echo "Creating SLURM job file..."
cat <<EOF > $JOB_SCRIPT
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$OUTPUT_FILE
#SBATCH --error=$ERROR_FILE
#SBATCH --time=$TIME_LIMIT
#SBATCH --mem=$MEMORY
#SBATCH --cpus-per-task=$CPUS
#SBATCH --partition=$PARTITION

# Load Python module
module load $PYTHON_MODULE

# Install required packages if needed
pip3 install --user pandas numpy scikit-learn matplotlib seaborn adjustText  # stabl commented out due to long runtime

echo "🚀 Starting COMPREHENSIVE serum biomarkers analysis..."
echo "📊 Using configuration from src/config.py"
echo "🔧 N_REPEATS: 100, TEST_SIZE: 0.2, PRETERM_CUTOFF: 37"
echo "🤖 Model Types: Lasso CV, ElasticNet CV"  # STABL commented out due to long runtime

# Step 1: Run gestational age analysis (using config values)
echo "📊 Step 1: Running gestational age analysis..."
python3 main.py gestational_age
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

echo "🎉 ALL ANALYSES COMPLETED at \$(date)"
echo "📊 Configuration used:"
echo "  - N_REPEATS: 100"
echo "  - TEST_SIZE: 0.2" 
echo "  - PRETERM_CUTOFF: 37"
EOF

# ----------------------------------------
# 🚀 STEP 2: Upload files
# ----------------------------------------
echo "Uploading ALL files to Sherlock..."
ssh $USER@$HOST "mkdir -p $REMOTE_DIR"
scp -r main.py src/ merge_all_results.py generate_comparison_plots.py organize_plots.py $USER@$HOST:$REMOTE_DIR/
scp -r analyze_hyperparameters.py comprehensive_hyperparameter_summary.py pipeline_summary.py plot_summary.py $USER@$HOST:$REMOTE_DIR/
scp -r data/ requirements.txt README.md $USER@$HOST:$REMOTE_DIR/
scp $JOB_SCRIPT $USER@$HOST:$REMOTE_DIR/

# ----------------------------------------
# 📤 STEP 3: Submit job
# ----------------------------------------
echo "Submitting comprehensive job..."
JOB_ID=$(ssh $USER@$HOST "cd $REMOTE_DIR && sbatch $JOB_SCRIPT" | awk '{print $4}')
echo "Submitted as Job ID: $JOB_ID"

# ----------------------------------------
# ⏳ STEP 4: Monitor job status
# ----------------------------------------
echo "Waiting for comprehensive job to complete..."
while true; do
  STATUS=$(ssh $USER@$HOST "squeue -j $JOB_ID -h -o '%T'")
  if [[ "$STATUS" == "" ]]; then
    echo "✅ Job $JOB_ID finished."
    break
  else
    echo "⏳ Job $JOB_ID is still $STATUS..."
    sleep 60
  fi
done

# ----------------------------------------
# 📥 STEP 5: Download results
# ----------------------------------------
echo "Downloading ALL result files..."
scp $USER@$HOST:$REMOTE_DIR/output_${JOB_ID}.txt .
scp $USER@$HOST:$REMOTE_DIR/error_${JOB_ID}.txt .
scp -r $USER@$HOST:$REMOTE_DIR/outputs/ ./outputs_sherlock_complete/
scp $USER@$HOST:$REMOTE_DIR/all_results*.pkl ./
scp $USER@$HOST:$REMOTE_DIR/*.log ./

echo "🎉 COMPREHENSIVE ANALYSIS COMPLETED!"
echo "📁 Results downloaded to ./outputs_sherlock_complete/"
echo "📊 Data files downloaded: all_results*.pkl"
echo "📋 Log files downloaded: *.log"
echo "🔧 Configuration used: src/config.py values" 