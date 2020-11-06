#!/bin/sh
#PBS -P ACD109103
#PBS -N harmonic
#PBS -q ctest
#PBS -l select=1:ncpus=8:mpiprocs=8
#PBS -l place=scatter
#PBS -l walltime=00:01:00
#PBS -j n
module purge
module load intel/2018_u1
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

date
for ((np=1; np<9; np=np+1)); do
  mpirun -n $np ./harmonic 1000000 100;
done
mpirun -n 1 ./harmonicSeq 1000000 100
