import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import pytorch_lightning as pl

from x_transformers.x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import top_k, AutoregressiveWrapper
from graph_transformer_pytorch.graph_transformer_pytorch.graph_transformer_pytorch import GraphTransformer


class XTModel(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

        # encoder one-hot embedding
        self.node_emb = nn.Linear(config['node_embd'], config['n_embd'])
        self.edge_emb = nn.Linear(config['edge_embd'], config['n_embd'])

        # graph encoder
        self.encoder = GraphTransformer(
            dim=config['n_embd'],
            depth=config['n_layer'],
            with_feedforwards=True,
            gated_residual=True,
            accept_adjacency_matrix=True,
            # experimental features
            norm_edges = True,
            rel_pos_emb = True,
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
        
        self.save_hyperparameters()

    def compute_loss(self, logits, targets):
        return F.cross_entropy(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=-1)
    
    @torch.no_grad()
    def metrics(self, batch, logits):
        _, _, _, _, _, smiles, _ = batch
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

    def forward(self, batch):
        node_feats, positions, src_mask, adj_mat, dense_edges_feats, p, dec_mask = batch
        nodes, edges = self.encoder(self.node_emb(node_feats.float()), self.edge_emb(dense_edges_feats.float()), adj_mat=adj_mat, mask=src_mask)
        logits, loss = self.decoder(p, context = nodes, context_mask = src_mask)
        return logits, loss

    def training_step(self, batch, batch_idx):
        logits, loss = self(batch)
        character_mismatch, accuracy = self.metrics(batch, logits)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('train_char_mismatch', character_mismatch, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits, loss = self(batch)
        character_mismatch, accuracy = self.metrics(batch, logits)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('val_char_mismatch', character_mismatch, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        logits, loss = self(batch)
        character_mismatch, accuracy = self.metrics(batch, logits)

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('test_char_mismatch', character_mismatch, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('test_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        reactants, products, src_mask = batch
        sample = self.encoder.generate(products, reactants[:, :1], reactants.size(1), mask=src_mask)

        mask = (reactants[:, 1:] != self.config['pad_token_id']).float()
        total_chars = mask.abs().sum()
        
        wrong_chars_batch = (reactants[:, 1:] != sample[:, :-1]) * mask
        character_mismatch = wrong_chars_batch.abs().sum()/total_chars
        accuracy = (torch.sum(wrong_chars_batch, dim=-1) == 0).abs().sum()/reactants.size(0)

        return batch, sample, character_mismatch, accuracy
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config['learning_rate'], 
            betas=(self.config['beta1'], self.config['beta2']), 
            weight_decay=self.config['weight_decay'], 
            )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_epochs'], eta_min=self.config['learning_rate']/10)
        return [optimizer], [scheduler] 