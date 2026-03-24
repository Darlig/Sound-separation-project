"""
Learning Rate Scaling Callback for PyTorch Lightning

Handles LR scaling when resuming training with different batch size.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class LRScalingCallback(Callback):
    """
    Callback to scale learning rate when batch size changes.
    
    Args:
        old_batch_size: Batch size used in previous training phase
        new_batch_size: Batch size for current training phase
        scale_type: 'linear' or 'sqrt' scaling (default: 'linear')
    """
    
    def __init__(self, old_batch_size, new_batch_size, scale_type='linear'):
        super().__init__()
        self.old_batch_size = old_batch_size
        self.new_batch_size = new_batch_size
        self.scale_type = scale_type
        self.lr_scaled = False
        
    def on_train_start(self, trainer, pl_module):
        """Scale learning rate at the start of training if not already done."""
        if self.lr_scaled:
            return
            
        # Calculate scale factor
        if self.scale_type == 'linear':
            scale_factor = self.new_batch_size / self.old_batch_size
        elif self.scale_type == 'sqrt':
            scale_factor = (self.new_batch_size / self.old_batch_size) ** 0.5
        else:
            raise ValueError(f"Unknown scale_type: {self.scale_type}")
        
        # Get optimizer
        optimizers = trainer.optimizers
        if not optimizers:
            return
            
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = old_lr * scale_factor
                param_group['lr'] = new_lr
                param_group['initial_lr'] = new_lr
                
                print(f"[LRScalingCallback] Scaled LR: {old_lr:.6f} -> {new_lr:.6f} "
                      f"(factor: {scale_factor:.3f})")
        
        # Also update LR scheduler if exists
        if trainer.lr_schedulers:
            for scheduler_config in trainer.lr_schedulers:
                scheduler = scheduler_config['scheduler']
                # Update base_lrs for ReduceLROnPlateau and similar schedulers
                if hasattr(scheduler, 'base_lrs'):
                    scheduler.base_lrs = [
                        lr * scale_factor for lr in scheduler.base_lrs
                    ]
        
        self.lr_scaled = True


class WarmupCallback(Callback):
    """
    Learning rate warmup callback.
    
    Gradually increases learning rate from 0 to target over warmup_steps.
    Useful when scaling batch size to maintain training stability.
    """
    
    def __init__(self, warmup_epochs=1, target_lr=None):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.initial_lr_set = False
        self.base_lrs = None
        
    def on_train_epoch_start(self, trainer, pl_module):
        if self.initial_lr_set:
            return
            
        current_epoch = trainer.current_epoch
        
        # Store base learning rates
        optimizers = trainer.optimizers
        if optimizers and self.base_lrs is None:
            self.base_lrs = [
                pg['lr'] for optimizer in optimizers 
                for pg in optimizer.param_groups
            ]
            if self.target_lr is not None:
                # Normalize to target
                scale = self.target_lr / self.base_lrs[0]
                self.base_lrs = [lr * scale for lr in self.base_lrs]
        
        if current_epoch < self.warmup_epochs and self.base_lrs is not None:
            # Linear warmup
            warmup_factor = (current_epoch + 1) / self.warmup_epochs
            
            lr_idx = 0
            for optimizer in optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.base_lrs[lr_idx] * warmup_factor
                    lr_idx += 1
            
            print(f"[WarmupCallback] Epoch {current_epoch}: LR warmed up to "
                  f"{self.base_lrs[0] * warmup_factor:.6f}")
        
        if current_epoch == self.warmup_epochs - 1:
            self.initial_lr_set = True
