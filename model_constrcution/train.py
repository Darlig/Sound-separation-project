import os
import argparse
import torch
import h5py
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from DNN_models.Complex_MTASS_model import ComplexMTASSLightning
from DNN_models.Complex_MTASS import *
from DNN_models.Complex_MTASS_Solver import *

class HDF5Dataset(Dataset):
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

def main(args):
    pl.seed_everything(42)
    # Load dataset
    data_train = HDF5Dataset(args.train_h5)
    data_val = HDF5Dataset(args.val_h5)
    train_loader = DataLoader(data_train,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.n_workers,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(data_val,
                            batch_size=args.eval_batch_size,
                            shuffle=False,
                            num_workers=args.n_workers,
                            pin_memory=True,
                            drop_last=True)
    
    model = ComplexMTASSLightning(
        learning_rate=args.lr,
        model_class=Complex_MTASS,
        loss_class=Complex_MTASS_model,
        use_l1_loss=args.use_l1_loss
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.exp_dir, 'checkpoints'),
        filename="{epoch:04d}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
    )

    logger = TensorBoardLogger(args.exp_dir, name="runs")

    trainer = pl.Trainer(
        default_root_dir=args.exp_dir,
        devices=args.gpus if args.use_cuda else "auto",
        accelerator="gpu" if args.use_cuda else "cpu",
        benchmark=True,
        strategy="ddp",
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        gradient_clip_val=20.0 if args.gradient_clip else 0.0,
        precision='32',
        #log_every_n_steps=10,
    )

    ckpt_path = None
    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        print(f"Resuming from checkpoint: {args.resume_ckpt}")
        ckpt_path = args.resume_ckpt
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_dir', type=str, default='./model_parameters')
    parser.add_argument('--train_h5', type=str, default='/ssd2.m2/sound/VGGSound/imagebind/train_ready.h5')
    parser.add_argument('--val_h5', type=str, default='/ssd2.m2/sound/VGGSound/imagebind/dev_ready.h5')
    parser.add_argument('--resume_ckpt', type=str, default=None, help="Path to .ckpt to continue training")
    parser.add_argument("--gpus", nargs="+", type=int, help="e.g. --gpus 0 1 2")

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--gradient_clip', action='store_true', help="Enable gradient clipping")
    parser.add_argument('--use_l1_loss', action='store_true', help="Enable L1 loss in addition to MSE and SI-SDR")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                        help="Whether to use cuda")

    args = parser.parse_args()

    os.makedirs(args.exp_dir, exist_ok=True)
    
    main(args)
