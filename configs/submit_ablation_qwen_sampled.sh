#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:a40:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_ablation_qwen_sampled_%j.out
#SBATCH --error=slurm_ablation_qwen_sampled_%j.err

source ~/.bashrc
source /home/nfs/lsuryana/miniconda3/etc/profile.d/conda.sh
conda activate rq2

cd ~/hamiltonian-vs-drivelm
export HF_HOME=/tudelft.net/staff-umbrella/lsuryana/huggingface

python run_ablation_pilot_qwen_sampled.py
