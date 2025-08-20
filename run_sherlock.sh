#!/bin/bash

USER="dibadin"
REMOTE_DIR="/home/users/dibadin/serum_biomarkers"
JOB_SCRIPT="job.slurm"
JOB_NAME="serum_biomarkers_complete"
OUTPUT_FILE="output_%j.txt"
ERROR_FILE="error_%j.txt"
TIME_LIMIT="72:00:00"
MEMORY="64G"
CPUS=16
PARTITION="dev"  # changed from normal to dev
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

module load $PYTHON_MODULE

echo "🚀 Starting analysis..."

python3 main.py gestational_age
python3 main.py birth_weight
python3 merge_all_results.py
python3 generate_comparison_plots.py
python3 plot_scatter_plots.py
python3 best_model_scatter_plots.py
python3 subgroup_analysis.py gestational_age
python3 subgroup_analysis.py birth_weight
python3 organize_plots.py
python3 analyze_hyperparameters.py
python3 comprehensive_hyperparameter_summary.py
python3 pipeline_summary.py
python3 plot_summary.py

echo "🎉 ALL ANALYSES COMPLETED at \$(date)"
EOF

# Submit job
sbatch $JOB_SCRIPT