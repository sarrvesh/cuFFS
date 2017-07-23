#!/bin/bash -l
#SBATCH --ntasks-per-node 1
#SBATCH --mem=10gb
#SBATCH --gres gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
##SBATCH --mail-user=sarrvesh.ss@gmail.com

# Load the modules
module load cfitsio
module load cuda/8.0.61
module load gcc
module load valgrind
export PATH="${PATH}:/home/see041/libconfig/lib:/home/see041/libconfig/include"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/home/see041/libconfig/lib"

# Launch job
rm /data/see041/wsrt_sings/trial1*
cd /data/see041/wsrt_sings/
##nvprof --analysis-metrics -o rmsynth.nvprof /home/see041/RMSynth_GPU/rmsynthesis /home/see041/RMSynth_GPU/parsetFile 
##nvprof /home/see041/RMSynth_GPU/rmsynthesis /home/see041/RMSynth_GPU/parsetFile
##valgrind --tool=memcheck --leak-check=yes --track-origins=yes /home/see041/RMSynth_GPU/rmsynthesis /home/see041/RMSynth_GPU/parsetFile
/home/see041/RMSynth_GPU/rmsynthesis /home/see041/RMSynth_GPU/parsetFile
