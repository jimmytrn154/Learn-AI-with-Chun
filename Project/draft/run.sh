#!/bin/bash

mkdir -p checkpoints

MAMMOCLIP_WEIGHTS="/mnt/disk1/backup_user/minh.ntn/Mammo-CLIP/Pre-trained-checkpoints/b5-model-best-epoch-7.tar"
SHOTS=128
SPLIT_FILE="/mnt/disk1/backup_user/minh.ntn/MammoCLIP_ExperimentSetup_Fewshot/data/monte_carlo_split.csv"

echo "================================================="
echo "STARTING MONTE CARLO FEW-SHOT EXPERIMENTS ($SHOTS SHOTS)"
echo "================================================="

# Danh sách các Run ID bạn muốn chạy (Ví dụ 5 run đầu tiên)
RUNS=("run_0" "run_1" "run_2" "run_3" "run_4")

# VÒNG LẶP DUYỆT QUA TỪNG RUN
for RUN_ID in "${RUNS[@]}"
do
    echo "-------------------------------------------------"
    echo ">>> RUNNING SPLIT: $RUN_ID"
    echo "-------------------------------------------------"

    # TASK 1: BASELINE (Image Only + Linear Head)

    echo " [TASK 1] MammoCLIP Baseline"
    python train.py --task 1 --epochs 50 --batch_size 8 --lr 1e-3 --shots $SHOTS --mammoclip_path "$MAMMOCLIP_WEIGHTS" --split_file "$SPLIT_FILE" --run_id "$RUN_ID"

    # TASK 2: CLASS GRAPH (Image Only + Li et al. Graph)
    echo " [TASK 2] MammoCLIP + ClassGraphAdapter"
    python train.py --task 2 --epochs 50 --batch_size 8 --lr 1e-3 --shots $SHOTS --mammoclip_path "$MAMMOCLIP_WEIGHTS" --split_file "$SPLIT_FILE" --run_id "$RUN_ID"

    # TASK 3: MULTIMODAL GRAPH (Image + ClinicalBERT)
    echo "[TASK 3] MammoCLIP + BioClinicalBERT + MultimodalGraph"
    python train.py --task 3 --epochs 50 --batch_size 8 --lr 1e-3 --shots $SHOTS --mammoclip_path "$MAMMOCLIP_WEIGHTS" --split_file "$SPLIT_FILE" --run_id "$RUN_ID"
    
    echo ">>> Completed $RUN_ID"
done

echo "================================================="
echo "ALL MONTE CARLO RUNS COMPLETED."
echo "================================================="