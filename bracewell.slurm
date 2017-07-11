#!/bin/bash -l
#SBATCH --ntasks-per-node 1
#SBATCH --mem=1gb
#SBATCH --gres gpu:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
##SBATCH --mail-user=sarrvesh.ss@gmail.com

# Load the modules
module load cfitsio
module load cuda/8.0.61
module load gcc
export PATH="${PATH}:/home/see041/libconfig/lib:/home/see041/libconfig/include"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/home/see041/libconfig/lib"

# Launch job
rm /data/see041/wsrt_sings/trial1*
cd /data/see041/wsrt_sings/
/home/see041/RMSynth_GPU/rmsynthesis /home/see041/RMSynth_GPU/parsetFile 
