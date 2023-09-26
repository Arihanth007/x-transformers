import os
import gc
import argparse
from glob import glob
import pandas as pd
from math import ceil

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from x_transformers.x_transformers import TransformerWrapper, Encoder, Decoder
from mlm_pytorch.mlm_pytorch.mlm_pytorch import MLM
from x_transformers.autoregressive_wrapper import top_k, AutoregressiveWrapper

import Chemformer.molbart.util as util
from Chemformer.molbart.data.datasets import Uspto50
from Chemformer.molbart.data.datamodules import FineTuneReactionDataModule

parser = argparse.ArgumentParser(description='Retrosynthesis')

parser.add_argument('--block_size', type=int, default=512, help='block size')
parser.add_argument('--vocab_size', type=int, default=530, help='vocab size')
parser.add_argument('--n_layer', type=int, default=6, help='number of layers')
parser.add_argument('--n_head', type=int, default=8, help='number of heads')
parser.add_argument('--n_embd', type=int, default=512, help='embedding dimension')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
parser.add_argument('--bias', type=bool, default=False, help='whether to use bias in attention layer')
parser.add_argument('--use_pos_emb', type=bool, default=False, help='whether to use positional embeddings')
parser.add_argument('--use_rotary_emb', type=bool, default=True, help='whether to use rotary embeddings')
parser.add_argument('--use_rel_pos_emb', type=bool, default=False, help='whether to use relative positional embeddings')
parser.add_argument('--mask_prob', type=float, default=0.0, help='mask probability')

parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2')
parser.add_argument('--weight_decay', type=float, default=1e-1, help='weight decay')

parser.add_argument('--data_dir', type=str, default='data/', help='data directory')
parser.add_argument('--validate_every', type=int, default=500, help='train iterations')
parser.add_argument('--validate_for', type=int, default=100, help='validate iterations')
parser.add_argument('--generate_for', type=int, default=2, help='generate iterations')
parser.add_argument('--train', type=bool, default=False, help='whether to train the model')
parser.add_argument('--finetune', type=bool, default=False, help='whether to pretrain the model')
parser.add_argument('--grad_accum', type=int, default=4, help='gradient accumulation')

parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--is_compile', type=bool, default=False, help='whether to compile the model')
parser.add_argument('--task', type=str, default='uspto50', help='task')
parser.add_argument('--run', type=str, default='exp', help='run name')
parser.add_argument('--project', type=str, default='uspto50', help='project name')
parser.add_argument('--entity', type=str, default='retrosynthesis', help='entity name')
parser.add_argument('--save_dir', type=str, default='.', help='save directory')
parser.add_argument('--log', type=bool, default=False, help='whether to log')
parser.add_argument('--set_precision', type=bool, default=False, help='whether to set precision')
parser.add_argument('--device_ids', type=int, nargs='*', help='device ids')
parser.add_argument('--vocab_file', type=str, default='', help='vocab files')
parser.add_argument('--sub_task', type=str, default='dec', help='sub task')
parser.add_argument('--load_from', type=str, default='', help='load checkpoint from')

config = vars(parser.parse_args())
config['data_dir'] = config["data_dir"] + config["task"]


class Chemformer(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

        with open('Chemformer/bart_vocab_downstream.txt', 'r') as f:
            vocab = f.read().split('\n')
        self.token_encode = {v: k for k, v in enumerate(vocab)}
        self.token_decode = {k: v for k, v in enumerate(vocab)}

        # transformer
        self.encoder = TransformerWrapper(
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
            self.encoder,
            mask_token_id = config['mask_token_id'], # the token id reserved for masking
            pad_token_id = config['pad_token_id'],   # the token id for padding
            mask_prob = 0,     # masking probability for masked language modeling
            replace_prob = 0.90,  # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
            # mask_ignore_token_ids = config['mask_ignore_token_ids']  # other tokens to exclude from masking, include the [cls] and [sep] here
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
                # layer_dropout = 0.1,   # stochastic depth - dropout entire layer
                # attn_dropout = 0.1,    # dropout post-attention
                # ff_dropout = 0.1,      # feedforward dropout
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
        reactants = batch["decoder_input"].transpose(0, 1)
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

    def forward(self, batch):
        products  = batch["encoder_input"].transpose(0, 1)
        reactants = batch["decoder_input"].transpose(0, 1)
        src_mask  = products != self.config['pad_token_id']

        # prods  = [''.join([self.token_decode[t.item()] for t in prod if t.item() != self.config['pad_token_id']]) for prod in products]
        # reacts = [''.join([self.token_decode[t.item()] for t in react if t.item() != self.config['pad_token_id']]) for react in reactants]
        # for r, p in zip(reacts, prods):
        #     print(f"{r} -> {p}")

        enc_logits, enc, enc_loss = self.encoder(products, mask=src_mask, return_logits_and_embeddings=True)
        # enc = self.encoder(products, mask=src_mask, return_embeddings=True)
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
        scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=self.config['learning_rate'])
        
        return [optimizer], scheduler


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    # torch.cuda.empty_cache()
    # gc.collect()

    print("Building tokeniser...")
    tokeniser = util.load_tokeniser('Chemformer/bart_vocab_downstream.txt', 272)
    print("Finished tokeniser.")

    config['pad_token_id'] = tokeniser.vocab[tokeniser.pad_token]
    config['mask_token_id'] = tokeniser.vocab[tokeniser.mask_token]

    print("Reading dataset...")
    dataset = Uspto50('data/uspto50/uspto_50.pickle', 0.5, forward=False)
    print("Finished dataset.")

    print("Building data module...")
    dm = FineTuneReactionDataModule(
            dataset,
            tokeniser,
            config['batch_size'],
            config['block_size'],
            forward_pred=False,
            val_idxs=dataset.val_idxs,
            test_idxs=dataset.test_idxs,
            train_token_batch_size=None,
            num_buckets=24,
            unified_model=False,
        )
    num_available_cpus = len(os.sched_getaffinity(0))
    num_workers = num_available_cpus // 3
    dm._num_workers = num_workers
    print(f"Using {str(num_workers)} workers for data module.")
    print("Finished datamodule.")

    model = Chemformer(config)

    logger = WandbLogger(
        # entity=config['entity'],
        project=config['project'],
        name=config['run'],
        save_dir=config['save_dir'],
        mode='disabled' if not config['log'] else 'online',
        )

    trainer = pl.Trainer(
        accelerator='gpu', devices=[1, 2, 3], strategy='ddp_find_unused_parameters_True',
        max_epochs=-1, logger=logger,
        precision='bf16-mixed' if config['set_precision'] else '32-true',
        gradient_clip_val=0.5, gradient_clip_algorithm='norm',
        accumulate_grad_batches=config['grad_accum'],
        # callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=dm)
    