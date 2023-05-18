#!/bin/bash
#SBATCH --job-name=gmee
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=p100l:4
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
pip install --no-index --upgrade pip

# install required packages
pip install torch --no-index
pip install datasets --no-index
pip install transformers --no-index
pip install accelerate --no-index
pip install sentencepiece --no-index

# run python scripts
echo "running scripts"
cd ../experiments

model_name="flan_t5-xxl"

# split each file into their own task 
srun -n 1 python *out*.py $model_name &
srun -n 1 python *with_*.py $model_name &
wait

# run results
srun -n 1 -c 12 python res*.py $model_name

echo "Job finished with exit code $? at: `date`"
