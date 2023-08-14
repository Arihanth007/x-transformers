import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import pytorch_lightning as pl

from x_transformers import XTransformer


class XTModel(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = XTransformer(
            dim=config['n_embd'],
            enc_num_tokens=config['vocab_size'],
            enc_depth=config['n_layer'],
            enc_heads=config['n_head'],
            enc_max_seq_len=config['block_size'],
            dec_num_tokens=config['vocab_size'],
            dec_depth=config['n_layer'],
            dec_heads=config['n_head'],
            dec_max_seq_len=config['block_size'],
            tie_token_emb = True,  # tie embeddings of encoder and decoder
        )
        self.model = torch.compile(self.model) if config['is_compile'] else self.model
        
        self.save_hyperparameters()

    def compute_loss(self, logits, targets):
        return F.cross_entropy(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=-1)

    def forward(self, batch):
        src, tgt, src_mask = batch
        loss = self.model(src, tgt, mask=src_mask)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'], 
            betas=(self.config['beta1'], self.config['beta2']), 
            weight_decay=self.config['weight_decay'], 
            )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_epochs'], eta_min=self.config['learning_rate']/10)
        return [optimizer], [scheduler] 