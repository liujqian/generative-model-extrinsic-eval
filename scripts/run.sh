#!/bin/bash
#SBATCH --job-name=gmee
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --array=1-2
#SBATCH --mem-per-cpu=25G

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
cd ../code/experiments

# split each file into their own job
if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]
then
	python *out*.py "flan_t5-xxl"
fi

if  [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]
then
	python *with_*.py "flan_t5-xxl"
fi

echo "Job ${SLURM_ARRAY_TASK_ID} finished with exit code $? at: `date`"
