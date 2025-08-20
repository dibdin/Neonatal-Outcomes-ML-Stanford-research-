#!/bin/bash

USER="dibadin"
REMOTE_DIR="/home/users/dibadin/serum_biomarkers"
JOB_SCRIPT="job_parallel.slurm"
JOB_NAME="serum_biomarkers_parallel"
OUTPUT_FILE="output_%A_%a.txt"
ERROR_FILE="error_%A_%a.txt"
TIME_LIMIT="48:00:00"
MEMORY="32G"
CPUS=8
PARTITION="dev"
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
#SBATCH --array=1-2

module load $PYTHON_MODULE

echo "🚀 Starting parallel analysis - Job \${SLURM_ARRAY_TASK_ID}"

# Job 1: Gestational Age
if [ \${SLURM_ARRAY_TASK_ID} -eq 1 ]; then
    echo "📊 Running Gestational Age Pipeline..."
    python3 main.py gestational_age
    python3 subgroup_analysis.py gestational_age
    python3 merge_all_results.py
    python3 generate_comparison_plots.py
    python3 plot_scatter_plots.py
    python3 best_model_scatter_plots.py
    python3 organize_plots.py
    python3 analyze_hyperparameters.py
    python3 comprehensive_hyperparameter_summary.py
    python3 pipeline_summary.py
    python3 plot_summary.py
    echo "✅ Gestational Age completed at \$(date)"
fi

# Job 2: Birth Weight
if [ \${SLURM_ARRAY_TASK_ID} -eq 2 ]; then
    echo "📊 Running Birth Weight Pipeline..."
    python3 main.py birth_weight
    python3 subgroup_analysis.py birth_weight
    python3 merge_all_results.py
    python3 generate_comparison_plots.py
    python3 plot_scatter_plots.py
    python3 best_model_scatter_plots.py
    python3 organize_plots.py
    python3 analyze_hyperparameters.py
    python3 comprehensive_hyperparameter_summary.py
    python3 pipeline_summary.py
    python3 plot_summary.py
    echo "✅ Birth Weight completed at \$(date)"
fi

echo "🎉 Job \${SLURM_ARRAY_TASK_ID} completed at \$(date)"
EOF

# Submit job array
sbatch $JOB_SCRIPT
