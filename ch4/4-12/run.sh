#!/bin/sh
#PBS -P ACD109103
#PBS -N simpson
#PBS -q ctest
#PBS -l select=1:ncpus=24:mpiprocs=24
#PBS -l place=scatter
#PBS -l walltime=00:01:00
#PBS -j n
module purge
module load intel/2018_u1
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

date
for ((np=1; np<25; np=np+1)); do
  mpirun -n $np ./simpson;
done
mpirun -n 1 ./simpsonSeq