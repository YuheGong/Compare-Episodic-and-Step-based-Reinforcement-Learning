#!/bin/bash
#SBATCH -p gpu_8
# #SBATCH -A 
#SBATCH -J dde2
#SBATCH --array 0-4%10

# Please use the complete path details :
#SBATCH -D /home/yre/Desktop/KIT/masterthesis/Compare-Episodic-and-Step-based-Reinforcement-Learning
#SBATCH -o ./slurm/slurmlog/out_%A_%a.log
#SBATCH -e ./slurm/slurmlog/err_%A_%a.log

# Cluster Settings
#SBATCH -n 1         # Number of tasks
#SBATCH -c 1  # Number of cores per task
#SBATCH -t 2:0:00             # 1:00:00 Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes

#SBATCH --gres gpu:1
# -------------------------------

# Activate the virtualenv / conda environment



# Export Pythonpath


# Additional Instructions from CONFIG.yml


python3 cw2cmaes/cmaes_cw2.py cw2cmaes/dmp_dense_2.yml -j $SLURM_ARRAY_TASK_ID 

# THIS WAS BUILT FROM THE DEFAULLT SBATCH TEMPLATE