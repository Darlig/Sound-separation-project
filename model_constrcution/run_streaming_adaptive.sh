#!/bin/bash

# Adaptive Streaming Training Script
# Supports dynamic batch_size and GPU count changes during training
#
# Usage:
#   Phase 1 (epochs 1-8):  batch=24, gpus=6
#   Phase 2 (epochs 9+):   batch=64, gpus=8

set -e  # Exit on error

# === Configuration ===
EXP_DIR=experiments/streaming_multi_head_separation_2-5mix_3class_overlap_nosing
TRAIN_H5="/work107/luoxiaoxue/workspace/Complex-MTASSNet/processed_data/3class_nosing/train_ready.h5"
VAL_H5="/work107/luoxiaoxue/workspace/Complex-MTASSNet/processed_data/3class_nosing/valid_ready.h5"

# Phase 1 config (epochs 1-8)
PHASE1_EPOCHS=8
PHASE1_BATCH=24
PHASE1_GPUS="0 1 2 3 4 5"
PHASE1_LR=0.001  # Base learning rate

# Phase 2 config (epochs 9+)
PHASE2_BATCH=64
PHASE2_GPUS="0 1 2 3 4 5 6 7"
PHASE2_LR=0.00267  # Linear scaling: 0.001 * (64/24) ≈ 0.00267

# Checkpoint path
RESUME_CKPT="${EXP_DIR}/checkpoints/last.ckpt"

# Function to get current epoch from checkpoint
get_current_epoch() {
    local ckpt_path=$1
    if [ -f "$ckpt_path" ]; then
        # Extract epoch number from checkpoint filename or metadata
        # Try to load using Python
        python3 -c "
import torch
try:
    ckpt = torch.load('$ckpt_path', map_location='cpu')
    epoch = ckpt.get('epoch', 0)
    print(epoch)
except:
    print(0)
" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# === Phase 1: Training with batch=24, gpus=6 (Epochs 1-8) ===
echo "=============================================="
echo "Phase 1: Training with batch=$PHASE1_BATCH, gpus=6"
echo "Target epochs: 1-$PHASE1_EPOCHS"
echo "Learning rate: $PHASE1_LR"
echo "=============================================="

if [ -f "$RESUME_CKPT" ]; then
    CURRENT_EPOCH=$(get_current_epoch "$RESUME_CKPT")
    echo "Found checkpoint at epoch: $CURRENT_EPOCH"
    
    if [ "$CURRENT_EPOCH" -lt "$PHASE1_EPOCHS" ]; then
        echo "Resuming Phase 1 training..."
        python3 train_streaming.py $EXP_DIR \
            --resume_ckpt $RESUME_CKPT \
            --train_h5 "$TRAIN_H5" \
            --val_h5 "$VAL_H5" \
            --use_cuda \
            --gpus $PHASE1_GPUS \
            --epochs $PHASE1_EPOCHS \
            --batch_size $PHASE1_BATCH \
            --lr $PHASE1_LR
    else
        echo "Phase 1 completed (epoch >= $PHASE1_EPOCHS)"
    fi
else
    echo "Starting Phase 1 from scratch..."
    python3 train_streaming.py $EXP_DIR \
        --train_h5 "$TRAIN_H5" \
        --val_h5 "$VAL_H5" \
        --use_cuda \
        --gpus $PHASE1_GPUS \
        --epochs $PHASE1_EPOCHS \
        --batch_size $PHASE1_BATCH \
        --lr $PHASE1_LR
fi

# === Phase 2: Training with batch=64, gpus=8 (Epochs 9+) ===
echo ""
echo "=============================================="
echo "Phase 2: Training with batch=$PHASE2_BATCH, gpus=8"
echo "Starting from epoch: $((PHASE1_EPOCHS + 1))"
echo "Learning rate: $PHASE2_LR (linear scaled)"
echo "=============================================="

# Check if Phase 1 checkpoint exists
if [ ! -f "$RESUME_CKPT" ]; then
    echo "Error: Checkpoint not found after Phase 1!"
    exit 1
fi

# Update learning rate in checkpoint (learning rate scaling)
echo "Updating learning rate in checkpoint..."
python3 -c "
import torch
import sys

ckpt_path = '$RESUME_CKPT'
new_lr = $PHASE2_LR
old_batch = $PHASE1_BATCH
new_batch = $PHASE2_BATCH

try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # Update learning rate in optimizer
    if 'optimizer_states' in ckpt and len(ckpt['optimizer_states']) > 0:
        opt_state = ckpt['optimizer_states'][0]
        
        # Get current LR
        old_lr = opt_state['param_groups'][0].get('lr', $PHASE1_LR)
        print(f'Old LR in checkpoint: {old_lr}')
        
        # Linear scaling: lr_new = lr_old * (new_batch / old_batch)
        # Or use the configured PHASE2_LR
        scaled_lr = old_lr * (new_batch / old_batch)
        
        # Update param_groups
        for pg in opt_state['param_groups']:
            pg['lr'] = scaled_lr
            pg['initial_lr'] = scaled_lr
        
        print(f'Updated LR to: {scaled_lr} (linear scaled)')
    
    # Save modified checkpoint
    torch.save(ckpt, ckpt_path)
    print('Checkpoint updated successfully!')
    
except Exception as e:
    print(f'Warning: Could not update checkpoint: {e}', file=sys.stderr)
    print('Continuing with original checkpoint...')
"

# Continue training with new configuration
echo "Starting Phase 2 with new configuration..."
python3 train_streaming.py $EXP_DIR \
    --resume_ckpt $RESUME_CKPT \
    --train_h5 "$TRAIN_H5" \
    --val_h5 "$VAL_H5" \
    --use_cuda \
    --gpus $PHASE2_GPUS \
    --epochs 100 \
    --batch_size $PHASE2_BATCH \
    --lr $PHASE2_LR

echo ""
echo "=============================================="
echo "Training completed!"
echo "=============================================="
