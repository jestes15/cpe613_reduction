#!/bin/bash
source /apps/profiles/modules_asax.sh.dyn
module load cuda
make A100 && ./main_a100