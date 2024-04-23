#!/bin/bash
source /apps/profiles/modules_asax.sh.dyn
module load cuda
make H100 && ./main_h100