#!/bin/bash
#SBATCH --job-name=gmee
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=2
#SBATCH --gres=gpu:v100l:4
#SBATCH --mem=64G

echo "Starting run at: `date`"

# activate modules
echo "adding modules"
module add python/3.9
module add StdEnv/2020 gcc/9.3.0 cuda/11.4
module add arrow/8.0.0
module add thrift/0.16.0

# activate env
echo "activatin venv"
virtualenv --download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install -q --no-index --upgrade pip

# install required packages
pip install torch -q --no-index
pip install datasets -q --no-index
pip install transformers -q --no-index
pip install accelerate -q --no-index
pip install sentencepiece -q --no-index
pip install setuptools -q --no-index
pip install wheel -q --no-index
pip install spacy -q 
python -m spacy download en_core_web_lg

# Log GPU usage
echo "gpus: `nvidia-smi -L`"

# Log memory usage every 30 seconds
log_memory_usage() {
	while true; do
		echo "GPU Memory Usage at $(date):"
		nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | awk -F',' '{ printf("  - GPU: %s\n    Memory Used: %s MiB\n    Memory Total: %s MiB\n    Memory Usage: %.2f%%\n", $1, $2, $3, ($2/$3)*100) }'
		sleep 600
	done
}

# Run the log_memory_usage function in the background
log_memory_usage &

# Set CUDA device IDs to be used
CUDA_VISIBLE_DEVICES=0,1,2,3

# run python scripts
echo "running scripts"
cd ../experiments

model_name="flan_t5-xxl"
# split each file into their own task 

srun -n 1 --gpus-per-task=2 accelerate launch *out*.py $model_name --num_processes 2 &
srun -n 1 --gpus-per-task=2 accelerate launch *with_*.py $model_name --num_processes 2 &
wait

# Stop logging memory usage
kill %1

# run results
srun -n 1 -c 24 accelerate launch res*.py $model_name --num_processes 2

echo "Job finished with exit code $? at: `date`"
