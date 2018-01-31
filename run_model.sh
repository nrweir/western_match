#!/bin/bash

#SBATCH -p gpu_requeue  ## partition: must be requesting GPUs to use this
#SBATCH -n 1  ## Number of CPU cores
#SBATCH --gres=gpu:1  ## Number of GPU cores requested
#SBATCH --constraint=cuda-7.5  ## make sure the right version of CUDA present
#SBATCH -t 0-4:00
#SBATCH --mem=10000  ## request 10 GB memory
#SBATCH -o %j.out  ## stdout will be written to [job number].out
#SBATCH -e %j.err  ## any error messages will be written to [job number].err

# load required modules
module load Anaconda3/4.3.0-fasrc01 gcc/4.8.2-fasrc01 cudnn/7.0-fasrc02 cuda/7.5-fasrc02 tensorflow/1.3.0-fasrc01

source activate /n/garner_lab/lab/python_envs/deepscope
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.5/extras/CUPTI/lib64  # until rc fixes this

python3 run_model.py 
