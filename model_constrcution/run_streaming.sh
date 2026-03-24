#!/bin/bash

# Streaming Complex MTASS Training Script
# Supports causal self-attention for streaming inference

exp_dir=experiments/streaming_multi_head_separation_2-5mix_3class_overlap_nosing
resume_ckpt="${exp_dir}/checkpoints/last.ckpt"

# Check if checkpoint exists
if [ -f "$resume_ckpt" ]; then
    echo "=============================================="
    echo "Resuming streaming training from checkpoint:"
    echo "$resume_ckpt"
    echo "=============================================="
    python3 train_streaming.py $exp_dir \
      --resume_ckpt $resume_ckpt \
      --train_h5 "/work107/duwenqiang/Sound-separation-project/processed_data//3class_nosing/train_ready.h5" \
      --val_h5 "/work107/duwenqiang/Sound-separation-project/processed_data//3class_nosing/valid_ready.h5" \
      --use_cuda \
      --gpus 0 1 2 3 4 5 \
      --batch_size 24 
else
    echo "=============================================="
    echo "Starting new streaming training..."
    echo "Model: StreamingComplexMTASS with Causal Self-Attention"
    echo "Checkpoint will be saved to: $exp_dir"
    echo "=============================================="
    python3 train_streaming.py $exp_dir \
      --train_h5 "/work107/duwenqiang/Sound-separation-project/processed_data//3class_nosing/train_ready.h5" \
      --val_h5 "/work107/duwenqiang/Sound-separation-project/processed_data//3class_nosing/valid_ready.h5" \
      --use_cuda \
      --gpus 0 1 2 3 4 5 \
      --batch_size 24
fi
