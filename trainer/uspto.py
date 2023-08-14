import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

import pytorch_lightning as pl

from model.uspto import GPTConfig, GPT


class NanoGPT(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = GPT(GPTConfig(
            block_size=config['block_size'],
            vocab_size=config['vocab_size'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_embd=config['n_embd'],
            dropout=config['dropout'],
            bias=config['bias'],
        ))
        self.model = torch.compile(self.model) if config['is_compile'] else self.model

        # token encoder and decoder
        with open(f'{config["data_dir"]}/vocab.txt', 'r') as f:
            self.token_decoder = f.read().splitlines()
        self.token_encoder = {k: v for v, k in enumerate(self.token_decoder)}
        
        self.save_hyperparameters()

    def compute_loss(self, logits, targets):
        return F.cross_entropy(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=self.token_encoder['<pad>'])

    def forward(self, batch):
        reactants, products = batch
        logits = self.model(reactants, products)
        loss = self.compute_loss(logits, reactants)
        return logits, loss

    def training_step(self, batch, batch_idx):
        _, loss = self(batch)
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        _, loss = self(batch)
        self.log('val_loss', loss, on_epoch=True, on_step=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        _, loss = self(batch)
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(
            learning_rate=self.config['learning_rate'], 
            betas=(self.config['beta1'], self.config['beta2']), 
            weight_decay=self.config['weight_decay'], 
            device_type='not_fused',
            )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_epochs'], eta_min=self.config['learning_rate']/10)
        return [optimizer], [scheduler] 