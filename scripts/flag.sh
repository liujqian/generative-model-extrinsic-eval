#!/bin/bash
#SBATCH --job-name=gmee
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --array=1-2
#SBATCH --mem-per-cpu=25G

while getopts "n:" opt; do
	case $opt in
		n)
			num=$OPTARG
			;;
		\?)
			echo "Invalid option: -$OPTARG" >&2
			exit 1
			;;
		:)
			echo "Option -$OPTARG requires an argument." >&2
			exit 1
			;;
	esac
done

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

# run python scripts
cd ../repo/experiments

if [ -z ${num+x} ]
then
	python *out*.py "mt0-xl"
	python *with_*.py "mt0-xl"
else
	if [ $num -eq 0 ]
	then
		python *out*.py "mt0-xl"
	elif [ $num -eq 1 ]
	then
		python *with_*.py "mt0-xl"
	fi
fi

echo "Job ${SLURM_ARRAY_TASK_ID} finished with exit code $? at: `date`"

