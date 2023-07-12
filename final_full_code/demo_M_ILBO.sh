# !/bin/bash

TRAINING_TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
TRAINING_LOG=${TRAINING_TIMESTAMP}.log
echo "$TRAINING_LOG"
bash run.sh | tee ./log/$TRAINING_LOG
