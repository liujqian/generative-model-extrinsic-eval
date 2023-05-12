#!/bin/bash
#SBATCH --job-name=gmee
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-cpu=20G

echo "Starting run at: `date`"

# activate modules
module add python/3.9
module add StdEnv/2020 gcc/9.3.0 cuda/11.4
module add arrow/8.0.0
module add thrift/0.16.0

echo `module list`

# activate env
virtualenv --download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

# install required packages
pip install torch --no-index
pip install datasets --no-index
pip install transformers --no-index
pip install accelerate --no-index
pip install sentencepiece --no-index
pip install setuptools --no-index
pip install wheel --no-index
pip install spacy
python -m spacy download en_core_web_lg

# run python scripts
cd ../repo/experiments

python results-analysis.py "mt0-xl" 
