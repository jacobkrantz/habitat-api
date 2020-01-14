#!/bin/bash

# Note: the hash-dollar, "#$" is not a commented out line.
# Instead, this is how you specify job parameters.

# use current working directory for input and output
# default is to use the users home directory
#$ -cwd

# name this job
#$ -N gpu-check-hpc

# Email to send to if I'm doing something wrong
#$ -M krantzja@oregonstate.edu

# send stdout and stderror to this file
#$ -o gpu-check-hpc.txt
#$ -j y

# select queue: dgx2-2, dgx2-3, dgx2-4, dgx2-5, dgx2-6, dgxs2 all failed (1/1/20).
#   queue instance "dgx2-2@compute-dgx2-2.hpc.engr.oregonstate.edu" dropped because it is temporarily not available (similar reason #1)
#   queue instance "dgx2-6@compute-dgx2-6.hpc.engr.oregonstate.edu" dropped because it is overloaded
#   has no permission for cluster queue "dgx2-3", "dgx2-4", "dgx2-5"
#$ -q dgx2-1

# print date and time
echo "-------------------------"
date

# check directory:
echo "dir:"
pwd

# see where the job is being run
echo "hostname:"
hostname

# check NVIDIA usage
nvidia-smi
sleep 3
nvidia-smi
sleep 3
nvidia-smi

echo "Job Finished."
date
echo "-------------------------"
