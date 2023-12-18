#!/bin/bash

find jobs -type f -name *.slurm | sort | xargs -n 1 -I {} sbatch {}
