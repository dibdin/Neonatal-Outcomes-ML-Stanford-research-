#!/bin/bash

USER="dibadin"
REMOTE_DIR="/home/users/dibadin/serum_biomarkers"
JOB_SCRIPT="job_optimized.slurm"
JOB_NAME="serum_biomarkers_optimized"
OUTPUT_FILE="output_%A_%a.txt"
ERROR_FILE="error_%A_%a.txt"
TIME_LIMIT="36:00:00"
MEMORY="64G"
CPUS=16
PARTITION="normal"  # Use normal partition for longer jobs
PYTHON_MODULE="python/3.9"

cat <<EOF > $JOB_SCRIPT
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$OUTPUT_FILE
#SBATCH --error=$ERROR_FILE
#SBATCH --time=$TIME_LIMIT
#SBATCH --mem=$MEMORY
#SBATCH --cpus-per-task=$CPUS
#SBATCH --partition=$PARTITION
#SBATCH --array=1-4

module load $PYTHON_MODULE

echo "🚀 Starting optimized parallel analysis - Job \${SLURM_ARRAY_TASK_ID}"

# Set environment variables for better performance
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=\$SLURM_CPUS_PER_TASK

# Job 1: Gestational Age - Both Samples
if [ \${SLURM_ARRAY_TASK_ID} -eq 1 ]; then
    echo "📊 Running Gestational Age - Both Samples..."
    python3 main.py gestational_age
    echo "✅ Gestational Age completed at \$(date)"
fi

# Job 2: Birth Weight - Both Samples  
if [ \${SLURM_ARRAY_TASK_ID} -eq 2 ]; then
    echo "📊 Running Birth Weight - Both Samples..."
    python3 main.py birth_weight
    echo "✅ Birth Weight completed at \$(date)"
fi

# Job 3: Subgroup Analysis - Gestational Age
if [ \${SLURM_ARRAY_TASK_ID} -eq 3 ]; then
    echo "📊 Running Subgroup Analysis - Gestational Age..."
    python3 subgroup_analysis.py gestational_age
    echo "✅ Gestational Age subgroup analysis completed at \$(date)"
fi

# Job 4: Subgroup Analysis - Birth Weight
if [ \${SLURM_ARRAY_TASK_ID} -eq 4 ]; then
    echo "📊 Running Subgroup Analysis - Birth Weight..."
    python3 subgroup_analysis.py birth_weight
    echo "✅ Birth Weight subgroup analysis completed at \$(date)"
fi

echo "🎉 Job \${SLURM_ARRAY_TASK_ID} completed at \$(date)"
EOF

# Submit job array
sbatch $JOB_SCRIPT
