#!/bin/bash
#SBATCH --job-name=noisy_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:30:00
#SBATCH --output=noisy_benchmark_COPD_snr_INFdB__%j.out
set -x # Enable script debugging
 
echo "Loading module 2023..."
module load 2023
 
echo "Loading CUDA/12.1.1..."
module load CUDA/12.1.1
module load cudnn/8.6 # Ensure cuDNN module is loaded if separate
 
echo "Updating LD_LIBRARY_PATH..."
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$HOME/miniconda3/lib:$LD_LIBRARY_PATH
 
echo "Activating virtual environment..."
source /home/milerisoy/miniconda3/bin/activate audio
 
echo "Verifying Python and CUDA..."
which python
python --version
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Devices:', torch.cuda.device_count())"
 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

task=$1
is_noisy=$2

echo "Starting evaluation script..."
sh scripts/eval_${task}.sh ${is_noisy}