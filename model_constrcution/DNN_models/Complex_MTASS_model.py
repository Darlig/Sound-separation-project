import torch
import pytorch_lightning as pl
from DNN_models.Complex_MTASS import *
from DNN_models.Complex_MTASS_Solver import *

class ComplexMTASSLightning(pl.LightningModule):
    def __init__(self, learning_rate, model_class, loss_class):
        super().__init__()
        self.save_hyperparameters(ignore=['model_class', 'loss_class'])
        self.model = model_class()
        self.loss_wrapper = loss_class
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X1 = batch[0]
        Y_targets = batch[1:4]
        R_targets = batch[4:7]

        Z1, Z2, Z3 = self(X1)

        loss, mse_loss, sisdr_loss = self.loss_wrapper.compute_out_cost(Z1, Z2, Z3, Y_targets, R_targets)
        #loss = self.loss_wrapper.compute_out_cost(Z1, Z2, Z3, Y_targets, R_targets)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('mse_loss', mse_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('sisdr_loss', -sisdr_loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        X1 = batch[0]
        Y_targets = batch[1:4]
        R_targets = batch[4:7]
        
        with torch.no_grad():
            Z1, Z2, Z3 = self(X1)
            loss, mse_loss, sisdr_loss = self.loss_wrapper.compute_out_cost(Z1, Z2, Z3, Y_targets, R_targets)
            #loss = self.loss_wrapper.compute_out_cost(Z1, Z2, Z3, Y_targets, R_targets)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_mse_loss', mse_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val_sisdr_loss', -sisdr_loss, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                                                        min_lr=5e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": schedular,
                "interval": "epoch",
                "monitor": "val_loss"
            },
        }