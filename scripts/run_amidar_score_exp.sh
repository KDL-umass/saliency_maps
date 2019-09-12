#!/bin/bash
#
#SBATCH --job-name=saliency_amidar
#SBATCH --output=saliency_maps/scripts/output/res_%j.txt   				# output file
#SBATCH -e saliency_maps/scripts/output/res_%j.err        				# File to which STDERR will be written
#SBATCH --partition=titanx-long         	# Partition to submit to
#
#SBATCH --mem=64000                     	# Memory required in MB
#SBATCH --gres=gpu:1                    	# No. of required GPUs
#SBATCH --ntasks-per-node=5            	    # No. of cores required
#SBATCH --mem-per-cpu=20000             	# Memory in MB per cpu allocated

echo "SLURM_JOBID: " $SLURM_JOBID

echo "Start running experiments"

echo "PYTHONPATH $PYTHONPATH"
echo "LD_LIBRARY_PATH $LD_LIBRARY_PATH"
echo "LIBCTOYBOX $LIBCTOYBOX"

source ~/.bashrc ; source activate tf

python -m saliency_maps.experiments.plot_amidar_score_exp -s='perturbation' -n=1000

echo "Done"

hostname
sleep 1
exit