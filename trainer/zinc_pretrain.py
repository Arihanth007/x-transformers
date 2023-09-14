from typing import Any, Optional
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import pytorch_lightning as pl

from x_transformers.x_transformers import TransformerWrapper, Encoder, Decoder
from mlm_pytorch.mlm_pytorch.mlm_pytorch import MLM
from x_transformers.autoregressive_wrapper import top_k, AutoregressiveWrapper


class XTModel(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

        # transformer
        encoder = TransformerWrapper(
            num_tokens=config['vocab_size'],
            max_seq_len=config['block_size'],
            attn_layers=Encoder(
                dim=config['n_embd'],
                depth=config['n_layer'],
                heads=config['n_head'],
                use_abs_pos_emb = config['use_pos_emb'],
                rotary_pos_emb = config['use_rotary_emb'],
                rel_pos_bias = config['use_rel_pos_emb'],
                ff_glu = True,
                ff_no_bias = True,
            ),
        )

        # masked language model
        self.encoder = MLM(
            encoder,
            mask_token_id = config['mask_token_id'], # the token id reserved for masking
            pad_token_id = config['pad_token_id'],   # the token id for padding
            mask_prob = 0.15,     # masking probability for masked language modeling
            replace_prob = 0.90,  # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
            mask_ignore_token_ids = config['mask_ignore_token_ids']  # other tokens to exclude from masking, include the [cls] and [sep] here
        )

        # transformer
        decoder = TransformerWrapper(
            num_tokens=config['vocab_size'],
            max_seq_len=config['block_size'],
            attn_layers=Decoder(
                dim=config['n_embd'],
                depth=config['n_layer'],
                heads=config['n_head'],
                use_abs_pos_emb = config['use_pos_emb'],
                rotary_pos_emb = config['use_rotary_emb'],
                rel_pos_bias = config['use_rel_pos_emb'],
                cross_attend = True,
                ff_glu = True,
                ff_no_bias = True,
                cross_residual_attn = True,
                shift_tokens = 1,
            ),
        )

        # auto-regressive decoder
        self.decoder = AutoregressiveWrapper(
            decoder,
            ignore_index=config['pad_token_id'],
            pad_value=config['pad_token_id'],
            mask_prob=config['mask_prob'],
        )

        # dont compile
        self.encoder = torch.compile(self.encoder) if config['is_compile'] else self.encoder
        self.decoder = torch.compile(self.decoder) if config['is_compile'] else self.decoder

        # number of epochs to train the only encoder for
        self.enc_epochs = config['enc_epochs']
        
        self.save_hyperparameters()

    def compute_loss(self, logits, targets):
        return F.cross_entropy(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=-1)
    
    @torch.no_grad()
    def metrics(self, batch, logits):
        smiles, _ = batch
        tgt = smiles[:, 1:]
        b, t, v = logits.size()
        
        logits = logits.contiguous().reshape(b*t, v)
        probs = F.softmax(top_k(logits), dim=-1)
        sample = torch.multinomial(probs, 1)
        sample = sample.contiguous().reshape(b, -1)
        
        mask = (tgt != self.config['pad_token_id']).float()
        total_chars = mask.abs().sum()
        
        wrong_chars_batch = ((tgt != sample) * mask).int()
        character_mismatch = wrong_chars_batch.sum()/total_chars
        accuracy = (torch.sum(wrong_chars_batch, dim=-1) == 0).sum()/b
        
        return character_mismatch, accuracy

    def forward(self, batch, task='enc'):
        smiles, src_mask = batch
        enc_logits, enc, enc_loss = self.encoder(smiles, mask=src_mask, return_logits_and_embeddings=True)
        if task == 'dec':
            dec_logits, dec_loss = self.decoder(smiles, context=enc, context_mask=src_mask)
            return dec_logits, dec_loss
        return enc_logits, enc_loss

    def training_step(self, batch, batch_idx):
        logits, loss = self(batch, task='enc' if self.current_epoch < self.enc_epochs else 'dec')
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        
        if self.current_epoch >= self.enc_epochs:
            character_mismatch, accuracy = self.metrics(batch, logits)
            self.log('train_char_mismatch', character_mismatch, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
            self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits, loss = self(batch, task='enc' if self.current_epoch < self.enc_epochs else 'dec')
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)

        if self.current_epoch >= self.enc_epochs:
            character_mismatch, accuracy = self.metrics(batch, logits)
            self.log('val_char_mismatch', character_mismatch, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
            self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config['learning_rate'], 
            betas=(self.config['beta1'], self.config['beta2']), 
            weight_decay=self.config['weight_decay'], 
            )
        
        num_batches = self.config['num_batches'] // self.trainer.accumulate_grad_batches
        self.print(f'num_batches: {num_batches}')
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.enc_epochs*num_batches, eta_min=self.config['learning_rate']/10)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step', # or 'epoch'
            'frequency': 1
            }
        
        return [optimizer], scheduler
