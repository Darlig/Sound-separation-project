"""
Adaptive Training Script for Streaming Complex MTASS
Supports dynamic batch size and GPU count changes with learning rate scaling.
"""

import os
import argparse
import torch
import h5py
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger

from DNN_models.Complex_MTASS_model_Streaming import ComplexMTASSStreamingLightning
from DNN_models.Complex_MTASS_Streaming import *
from DNN_models.Complex_MTASS_Solver_Streaming import *
from DNN_models.lr_scaling_callback import LRScalingCallback, WarmupCallback


class HDF5Dataset:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file = None
        with h5py.File(h5_path, 'r') as f:
            self.length = f['X1'].shape[0]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            
        X1 = torch.from_numpy(self.h5_file['X1'][idx]).float()
        Y_targets = [torch.from_numpy(self.h5_file[f'Y{i}'][idx]).float() for i in range(1, 4)]
        R_targets = [torch.from_numpy(self.h5_file[f'R{i}'][idx]).float() for i in range(1, 4)]
        
        return (X1, *Y_targets, *R_targets)
    
    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()


class ConfigMonitorCallback(Callback):
    """Monitor and log training configuration changes."""
    
    def __init__(self):
        super().__init__()
        self.logged = False
    
    def on_train_start(self, trainer, pl_module):
        if not self.logged:
            print("\n" + "="*60)
            print("Training Configuration:")
            print("="*60)
            print(f"  Batch size: {trainer.train_dataloader.batch_size}")
            print(f"  GPUs: {trainer.num_devices}")
            print(f"  World size: {trainer.world_size}")
            print(f"  Effective batch size: {trainer.train_dataloader.batch_size * trainer.world_size}")
            print(f"  Current epoch: {trainer.current_epoch}")
            print(f"  Learning rate: {pl_module.learning_rate}")
            print("="*60 + "\n")
            self.logged = True


def main(args):
    pl.seed_everything(42)
    
    print("="*60)
    print("Adaptive Streaming Complex MTASS Training")
    print("="*60)
    print(f"Phase: {args.phase}")
    print(f"Batch size: {args.batch_size} (per GPU)")
    print(f"GPUs: {args.gpus}")
    print(f"Effective batch size: {args.batch_size * len(args.gpus)}")
    print(f"Learning rate: {args.lr}")
    if args.old_batch_size and args.old_batch_size != args.batch_size:
        scale_factor = args.batch_size / args.old_batch_size
        print(f"LR scaling factor: {scale_factor:.3f} (from batch {args.old_batch_size})")
    print("="*60)
    
    # Load dataset
    data_train = HDF5Dataset(args.train_h5)
    data_val = HDF5Dataset(args.val_h5)
    
    train_loader = DataLoader(
        data_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        data_val,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Setup callbacks
    callbacks = []
    
    # Learning rate scaling if batch size changed
    if args.old_batch_size and args.old_batch_size != args.batch_size:
        lr_scale_callback = LRScalingCallback(
            old_batch_size=args.old_batch_size,
            new_batch_size=args.batch_size,
            scale_type='linear'
        )
        callbacks.append(lr_scale_callback)
        
        # Add warmup for stability after scaling
        if args.warmup_epochs > 0:
            warmup_callback = WarmupCallback(warmup_epochs=args.warmup_epochs)
            callbacks.append(warmup_callback)
    
    # Config monitor
    callbacks.append(ConfigMonitorCallback())
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.exp_dir, 'checkpoints'),
        filename="{epoch:04d}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Logger
    logger = TensorBoardLogger(args.exp_dir, name="runs")
    
    # Model
    model = ComplexMTASSStreamingLightning(
        learning_rate=args.lr,
        model_class=StreamingComplexMTASS,
        loss_class=Complex_MTASS_model_Streaming
    )
    
    # Trainer
    trainer = pl.Trainer(
        default_root_dir=args.exp_dir,
        devices=args.gpus if args.use_cuda else "auto",
        accelerator="gpu" if args.use_cuda else "cpu",
        benchmark=True,
        strategy="ddp" if len(args.gpus) > 1 else "auto",
        max_epochs=args.epochs,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=20.0 if args.gradient_clip else 0.0,
        precision='32',
    )
    
    # Resume from checkpoint if provided
    ckpt_path = None
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        print(f"Resuming from checkpoint: {args.resume_ckpt}")
        ckpt_path = args.resume_ckpt
    
    # Train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    
    print("\n" + "="*60)
    print(f"Phase {args.phase} completed!")
    print(f"Checkpoint saved to: {os.path.join(args.exp_dir, 'checkpoints')}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adaptive Training for Streaming Complex MTASS')
    
    # Basic arguments
    parser.add_argument('exp_dir', type=str, default='./model_parameters_streaming')
    parser.add_argument('--train_h5', type=str, required=True)
    parser.add_argument('--val_h5', type=str, required=True)
    parser.add_argument('--resume_ckpt', type=str, default=None)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    
    # Training config
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--gradient_clip', action='store_true')
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true')
    
    # Phase and LR scaling
    parser.add_argument('--phase', type=str, default='1', help='Training phase identifier')
    parser.add_argument('--old_batch_size', type=int, default=None, 
                       help='Previous batch size for LR scaling')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                       help='Number of warmup epochs after LR scaling')
    
    args = parser.parse_args()
    
    os.makedirs(args.exp_dir, exist_ok=True)
    
    main(args)
