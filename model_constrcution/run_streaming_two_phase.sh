#!/bin/bash

# Two-Phase Streaming Training Script
# Phase 1: batch=24, gpus=6, epochs=1-8
# Phase 2: batch=64, gpus=8, epochs=9+
#
# Handles learning rate scaling automatically when switching phases

set -e

# === Configuration ===
EXP_DIR=experiments/streaming_multi_head_separation_2-5mix_3class_overlap_nosing
TRAIN_H5="/work107/luoxiaoxue/workspace/Complex-MTASSNet/processed_data/3class_nosing/train_ready.h5"
VAL_H5="/work107/luoxiaoxue/workspace/Complex-MTASSNet/processed_data/3class_nosing/valid_ready.h5"

CKPT_PATH="${EXP_DIR}/checkpoints/last.ckpt"

# Phase 1 Configuration
P1_EPOCHS=8
P1_BATCH=24
P1_GPUS=(0 1 2 3 4 5)
P1_LR=0.001

# Phase 2 Configuration  
P2_BATCH=64
P2_GPUS=(0 1 2 3 4 5 6 7)
P2_LR=0.001  # Base LR, will be scaled automatically

# Helper function to print section headers
print_header() {
    echo ""
    echo "=============================================="
    echo "$1"
    echo "=============================================="
}

# Helper function to get current epoch from checkpoint
get_current_epoch() {
    if [ -f "$CKPT_PATH" ]; then
        python3 -c "
import torch
try:
    ckpt = torch.load('$CKPT_PATH', map_location='cpu')
    epoch = ckpt.get('epoch', 0)
    print(epoch)
except:
    print(0)
" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# ============================================
# PHASE 1: Initial Training (Epochs 1-8)
# ============================================
print_header "PHASE 1: Batch=$P1_BATCH, GPUs=${#P1_GPUS[@]}, Epochs=1-$P1_EPOCHS"

CURRENT_EPOCH=$(get_current_epoch)
print_header "Current checkpoint epoch: $CURRENT_EPOCH"

if [ "$CURRENT_EPOCH" -lt "$P1_EPOCHS" ]; then
    echo "Starting/Resuming Phase 1..."
    
    if [ -f "$CKPT_PATH" ]; then
        # Resume from checkpoint
        python3 train_streaming_adaptive.py $EXP_DIR \
            --resume_ckpt $CKPT_PATH \
            --train_h5 "$TRAIN_H5" \
            --val_h5 "$VAL_H5" \
            --gpus "${P1_GPUS[@]}" \
            --epochs $P1_EPOCHS \
            --batch_size $P1_BATCH \
            --eval_batch_size $P1_BATCH \
            --lr $P1_LR \
            --phase "1" \
            --use_cuda \
            --n_workers 8
    else
        # Start from scratch
        python3 train_streaming_adaptive.py $EXP_DIR \
            --train_h5 "$TRAIN_H5" \
            --val_h5 "$VAL_H5" \
            --gpus "${P1_GPUS[@]}" \
            --epochs $P1_EPOCHS \
            --batch_size $P1_BATCH \
            --eval_batch_size $P1_BATCH \
            --lr $P1_LR \
            --phase "1" \
            --use_cuda \
            --n_workers 8
    fi
    
    echo "Phase 1 completed!"
else
    echo "Phase 1 already completed (epoch $CURRENT_EPOCH >= $P1_EPOCHS)"
fi

# ============================================
# PHASE 2: Scaled Training (Epochs 9+)
# ============================================
print_header "PHASE 2: Batch=$P2_BATCH, GPUs=${#P2_GPUS[@]}, Epochs=$P1_EPOCHS+"

# Verify Phase 1 checkpoint exists
if [ ! -f "$CKPT_PATH" ]; then
    print_header "ERROR: Checkpoint not found!"
    echo "Expected: $CKPT_PATH"
    exit 1
fi

# Calculate learning rate scaling
LR_SCALE=$(echo "scale=4; $P2_BATCH / $P1_BATCH" | bc)
SCALED_LR=$(echo "scale=6; $P1_LR * $LR_SCALE" | bc)

echo "Learning Rate Scaling:"
echo "  Old batch size: $P1_BATCH"
echo "  New batch size: $P2_BATCH"
echo "  Scale factor: $LR_SCALE"
echo "  Base LR: $P1_LR"
echo "  Scaled LR: $SCALED_LR"
echo ""
echo "Starting Phase 2 with automatic LR scaling..."

python3 train_streaming_adaptive.py $EXP_DIR \
    --resume_ckpt $CKPT_PATH \
    --train_h5 "$TRAIN_H5" \
    --val_h5 "$VAL_H5" \
    --gpus "${P2_GPUS[@]}" \
    --epochs 100 \
    --batch_size $P2_BATCH \
    --eval_batch_size $P2_BATCH \
    --lr $P2_LR \
    --old_batch_size $P1_BATCH \
    --warmup_epochs 2 \
    --phase "2" \
    --use_cuda \
    --n_workers 8 \
    --gradient_clip

print_header "Training Completed!"
echo "Final checkpoint: $CKPT_PATH"
echo ""
echo "To test the model:"
echo "  python test_streaming.py --ckpt_path $CKPT_PATH --test_h5 <test.h5> --num_sources 3"
