#!/bin/bash

# Note: the hash-dollar, "#$" is not a commented out line.
# Instead, this is how you specify job parameters.

# use current working directory for input and output
# default is to use the users home directory
#$ -cwd

# name this job
#$ -N habitat-api-0

# Email to send to if I'm doing something wrong
#$ -M krantzja@oregonstate.edu

# send stdout and stderror to this file
#$ -o habitat-api-0.txt
#$ -j y

# select queue: dgx2-2, dgx2-5, dgxs2 all failed (12/24/19).
#$ -q dgx2-1

# print date and time
echo "-------------------------"
date

# see where the job is being run
echo "hostname:"
hostname
nvidia-smi

# check directory:
echo "dir:"
pwd

# setup the environment
source ../habitat-sim/venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib64/nvidia/:/usr/local/common/gcc-6.3.0/lib64:/usr/local/common/gcc-6.3.0/lib:$LD_LIBRARY_PATH
pip install -r requirements.txt
pip install -r habitat_baselines/rl/requirements.txt
python setup.py develop --all

# job scripts to run
python habitat_baselines/run.py --run-type train --exp-config habitat_baselines/config/vln/imitation_vln.yaml

echo "Job Finished."
date
echo "-------------------------"
