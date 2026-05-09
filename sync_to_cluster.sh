#!/bin/bash
# Syncs specialist checkpoints (gitignored) to JED cluster before training.
# Usage: bash sync_to_cluster.sh <gaspar_username>

USER=${1:?"Usage: bash sync_to_cluster.sh <gaspar_username>"}
REMOTE="${USER}@jed.hpc.epfl.ch:~/ER_course/micro-515-EvoRob/"

echo "Syncing checkpoints to ${REMOTE}..."

rsync -avz --progress \
    results/best_flat/ \
    "${REMOTE}results/best_flat/"

rsync -avz --progress \
    results/best_hill/ \
    "${REMOTE}results/best_hill/"

rsync -avz --progress \
    results/20260324_085356_nsga_ckpts/ \
    "${REMOTE}results/20260324_085356_nsga_ckpts/"

echo "Done. Now on the cluster:"
echo "  cd ~/ER_course/micro-515-EvoRob"
echo "  sbatch run_final_project.sh"
