import rdkit
from rdkit import Chem

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import pytorch_lightning as pl

from x_transformers.x_transformers import TransformerWrapper, Encoder, Decoder
from mlm_pytorch.mlm_pytorch.mlm_pytorch import MLM
from x_transformers.autoregressive_wrapper import top_k, AutoregressiveWrapper

# ignore rdkit warnings
rdkit.RDLogger.DisableLog('rdApp.*')


class XTModel(pl.LightningModule):
    def __init__(self, config: dict, token_encoder: dict=None, token_decoder: list=None) -> None:
        super().__init__()
        self.config = config
        self.token_encoder = token_encoder
        self.token_decoder = token_decoder

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
                attn_flash = True if not config['use_rel_pos_emb'] else False,
            ),
        )

        # masked language model
        self.encoder = MLM(
            encoder,
            mask_token_id = config['mask_token_id'], # the token id reserved for masking
            pad_token_id = config['pad_token_id'],   # the token id for padding
            mask_prob = 0,     # masking probability for masked language modeling
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
                # cross_residual_attn = True,
                # shift_tokens = 1,
                attn_flash = True if not config['use_rel_pos_emb'] else False,
                layer_dropout = 0.1,   # stochastic depth - dropout entire layer
                attn_dropout = 0.1,    # dropout post-attention
                ff_dropout = 0.1,      # feedforward dropout
            ),
        )

        # auto-regressive decoder
        self.decoder = AutoregressiveWrapper(
            decoder,
            ignore_index=config['pad_token_id'],
            pad_value=config['pad_token_id'],
            # mask_prob=config['mask_prob'],
        )

        # dont compile
        self.encoder = torch.compile(self.encoder) if config['is_compile'] else self.encoder
        self.decoder = torch.compile(self.decoder) if config['is_compile'] else self.decoder

        self.save_hyperparameters()

    def compute_loss(self, logits, targets):
        return F.cross_entropy(logits.contiguous().view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=-1)
    
    @torch.no_grad()
    def metrics(self, batch, logits):
        reactants, products, src_mask = batch
        tgt = reactants[:, 1:]
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
    
    @torch.no_grad()
    def test_metrics(self, batch, logits):
        reactants, products, src_mask = batch
        tgt = reactants[:, 1:]
        b, t, v = logits.size()
        
        logits = logits.contiguous().reshape(b*t, v)
        probs = F.softmax(top_k(logits), dim=-1)
        sample = torch.multinomial(probs, 1)
        sample = sample.contiguous().reshape(b, -1)
        
        decoded_targets = []
        for i, target in enumerate(tgt):
            decoded_targets.append([])
            for tok in target:
                if tok in self.config['mask_ignore_token_ids']:
                    break
                decoded_targets[i].append(self.token_decoder[tok])
        
        decoded_samples = []
        for i, pred_sample in enumerate(sample):
            decoded_samples.append([])
            for tok in pred_sample:
                if tok in self.config['mask_ignore_token_ids']:
                    break
                decoded_samples[i].append(self.token_decoder[tok])

        target_mols = []
        for i, target in enumerate(decoded_targets):
            try:
                target_mols.append(Chem.MolFromSmiles(''.join(target)))
            except:
                target_mols.append(None)
        predicted_mols = []
        for i, pred in enumerate(decoded_samples):
            try:
                predicted_mols.append(Chem.MolFromSmiles(''.join(pred)))
            except:
                predicted_mols.append(None)

        accuracy, invalid = 0, 0
        for i, (target_mol, pred_mol) in enumerate(zip(target_mols, predicted_mols)):
            if target_mol is None or pred_mol is None:
                invalid += 1
            elif target_mol.HasSubstructMatch(pred_mol) and pred_mol.HasSubstructMatch(target_mol):
                accuracy += 1
        
        return accuracy/len(target_mols), accuracy/(len(target_mols)-invalid) if len(target_mols) != invalid else 0. , invalid/len(target_mols)

    def forward(self, batch):
        reactants, products, src_mask = batch
        enc_logits, enc, enc_loss = self.encoder(products, mask=src_mask, return_logits_and_embeddings=True)
        dec_logits, dec_loss = self.decoder(reactants, context=enc, context_mask=src_mask)
        return dec_logits, dec_loss

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
        mol_accuracy, mol_adjusted_accuracy, mol_invalid = self.test_metrics(batch, logits)

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('test_char_mismatch', character_mismatch, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('test_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)

        self.log('test_mol_accuracy', mol_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('test_mol_adjusted_accuracy', mol_adjusted_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('test_mol_invalid', mol_invalid, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config['learning_rate'], 
            betas=(self.config['beta1'], self.config['beta2']), 
            weight_decay=self.config['weight_decay'], 
            )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_epochs'], eta_min=self.config['learning_rate']/10)
        
        return [optimizer], scheduler
