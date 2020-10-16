#!/bin/sh
#PBS -P ACD109103
#PBS -N reduction
#PBS -q ctest
#PBS -l select=1:ncpus=2:mpiprocs=10
#PBS -l place=scatter
#PBS -l walltime=00:01:00
#PBS -j n
module purge
module load intel/2018_u1
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

date
mpirun -n 100 ./reduction;