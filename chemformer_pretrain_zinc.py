import os
import gc
import math
import argparse
from glob import glob
from tqdm import tqdm
from rdkit import Chem
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LambdaLR
from warmup_scheduler import GradualWarmupScheduler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from x_transformers.x_transformers_lora import TransformerWrapper, Encoder, Decoder
from mlm_pytorch.mlm_pytorch.mlm_pytorch import MLM
from x_transformers.autoregressive_wrapper import top_k, top_a, top_p, AutoregressiveWrapper

import Chemformer.molbart.util as util
from Chemformer.molbart.data.datasets import Zinc
from Chemformer.molbart.data.datamodules import MoleculeDataModule

# disable rdkit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

parser = argparse.ArgumentParser(description='Retrosynthesis')

parser.add_argument('--block_size', type=int, default=512, help='block size')
parser.add_argument('--vocab_size', type=int, default=530, help='vocab size')
parser.add_argument('--n_layer', type=int, default=6, help='number of layers')
parser.add_argument('--n_head', type=int, default=8, help='number of heads')
parser.add_argument('--n_embd', type=int, default=512, help='embedding dimension')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
parser.add_argument('--bias', type=bool, default=False, help='whether to use bias in attention layer')
parser.add_argument('--use_pos_emb', type=bool, default=False, help='whether to use positional embeddings')
parser.add_argument('--use_rotary_emb', type=bool, default=False, help='whether to use rotary embeddings')
parser.add_argument('--use_rel_pos_emb', type=bool, default=False, help='whether to use relative positional embeddings')
parser.add_argument('--mask_prob', type=float, default=0.0, help='mask probability')

parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2')
parser.add_argument('--weight_decay', type=float, default=1e-1, help='weight decay')
parser.add_argument('--lr_scheduler', type=str, default='onecycle', help='lr scheduler')
parser.add_argument('--dividing_factor', type=float, default=10000, help='dividing factor for lr scheduler')
parser.add_argument('--warm_up_steps', type=int, default=8000, help='warm up steps')

parser.add_argument('--data_dir', type=str, default='data/', help='data directory')
parser.add_argument('--validate_every', type=int, default=500, help='train iterations')
parser.add_argument('--validate_for', type=int, default=100, help='validate iterations')
parser.add_argument('--generate_every', type=int, default=10, help='interval to generate')
parser.add_argument('--generate_for', type=int, default=2, help='generate iterations')
parser.add_argument('--train', type=bool, default=False, help='whether to train the model')
parser.add_argument('--finetune', type=bool, default=False, help='whether to pretrain the model')
parser.add_argument('--grad_accum', type=int, default=4, help='gradient accumulation')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers')

parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--is_compile', type=bool, default=False, help='whether to compile the model')
parser.add_argument('--task', type=str, default='uspto50', help='task')
parser.add_argument('--run', type=str, default='exp', help='run name')
parser.add_argument('--project', type=str, default='uspto50', help='project name')
parser.add_argument('--entity', type=str, default='retrosynthesis', help='entity name')
parser.add_argument('--save_dir', type=str, default='/scratch/arihanth.srikar', help='save directory')
parser.add_argument('--log', type=bool, default=False, help='whether to log')
parser.add_argument('--set_precision', type=bool, default=False, help='whether to set precision')
parser.add_argument('--device_ids', type=int, nargs='*', help='device ids')
parser.add_argument('--vocab_file', type=str, default='', help='vocab files')
parser.add_argument('--sub_task', type=str, default='dec', help='sub task')
parser.add_argument('--load_from', type=str, default='', help='load checkpoint from')

config = vars(parser.parse_args())
config['data_dir'] = config["data_dir"] + config["task"]


class FuncLR(LambdaLR):
    def get_lr(self):
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]


class Chemformer(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

        with open('Chemformer/my_vocab.txt', 'r') as f:
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
                layer_dropout = 0.1,   # stochastic depth - dropout entire layer
                # attn_flash = True if not config['use_rel_pos_emb'] else False,
            ),
        )

        # masked language model
        self.encoder = MLM(
            self.encoder,
            mask_token_id = config['mask_token_id'], # the token id reserved for masking
            pad_token_id = config['pad_token_id'],   # the token id for padding
            mask_prob = config['mask_prob'],     # masking probability for masked language modeling
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
                # shift_tokens = 1,
                # attn_flash = True if not config['use_rel_pos_emb'] else False,
                layer_dropout = 0.1,   # stochastic depth - dropout entire layer
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

        # pretraining (freeze lora, train rest)
        lora_params = 0
        for n, p in self.named_parameters():
            if 'lora_' in n:
                p.requires_grad = False
                lora_params += p.numel()
            else:
                p.requires_grad = True
        print(f"Freezing {lora_params/1e6:.2f}M parameters")

        self.save_hyperparameters()

    @torch.no_grad()
    def metrics(self, batch, logits):
        aug_smi = batch["decoder_input"].transpose(0, 1)
        tgt = aug_smi[:, 1:]
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
    def validate_metrics(self, batch):
        total_acc   = []
        partial_acc = []

        mol_smi = batch["encoder_input"].transpose(0, 1)
        aug_smi = batch["decoder_input"].transpose(0, 1)
        src_mask = mol_smi != self.config['pad_token_id']

        _, enc, _ = self.encoder(mol_smi, mask=src_mask, return_logits_and_embeddings=True)
        gen_seq = self.decoder.generate(aug_smi[:, :1], aug_smi.shape[1], context=enc, context_mask=src_mask, filter_logits_fn=top_k, filter_thres=0.999)

        mols = [''.join([self.token_decode[t.item()] for t in react if t.item() != self.config['pad_token_id']])[1:-1] for react in aug_smi]
        gens = [''.join([self.token_decode[t.item()] for t in gen if t.item() != self.config['pad_token_id']]) for gen in gen_seq]

        for i, (r, g) in enumerate(zip(mols, gens)):
            
            # find first occurence of '&' in g
            idx = g.find('&')
            g = g[:idx] if idx != -1 else g
            # print(f"{i}:{p}\n{r}\n{g}")

            r_mol = Chem.MolFromSmiles(r)
            g_mol = Chem.MolFromSmiles(g)
            if r_mol is None:
                continue
            total_acc.append(1 if g_mol is not None and r_mol.HasSubstructMatch(g_mol) and g_mol.HasSubstructMatch(r_mol) else 0)
            
            r_mols = [Chem.MolFromSmiles(r_smi) for r_smi in r.split('.')]
            g_mols = [Chem.MolFromSmiles(g_smi) for g_smi in g.split('.')]
            par_acc = 0
            for small_g in g_mols:
                for small_r in r_mols:
                    if small_g is not None and small_r is not None and small_r.HasSubstructMatch(small_g) and small_g.HasSubstructMatch(small_r):
                        par_acc = 1
            partial_acc.append(par_acc)
        
        return {'total_accuracy': sum(total_acc)/len(total_acc), 'partial_accuracy': sum(partial_acc)/len(partial_acc)}

    def forward(self, batch):
        mol_smi = batch["encoder_input"].transpose(0, 1)
        aug_smi = batch["decoder_input"].transpose(0, 1)
        src_mask = mol_smi != self.config['pad_token_id']

        enc_logits, enc, enc_loss = self.encoder(mol_smi, mask=src_mask, return_logits_and_embeddings=True)
        dec_logits, dec_loss = self.decoder(aug_smi, context=enc, context_mask=src_mask)
        
        return dec_logits, enc_loss, dec_loss

    def training_step(self, batch, batch_idx):
        logits, enc_loss, dec_loss = self(batch)
        character_mismatch, accuracy = self.metrics(batch, logits)
        loss = enc_loss + dec_loss

        # get learning rate
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        
        self.log('train_enc_loss', enc_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('train_dec_loss', dec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('train_char_mismatch', character_mismatch, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits, enc_loss, dec_loss = self(batch)
        character_mismatch, accuracy = self.metrics(batch, logits)
        loss = enc_loss + dec_loss

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('val_enc_loss', enc_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('val_dec_loss', dec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('val_char_mismatch', character_mismatch, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        logits, enc_loss, dec_loss = self(batch)
        character_mismatch, accuracy = self.metrics(batch, logits)
        my_dict = self.validate_metrics(batch)
        loss = enc_loss + dec_loss

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('test_enc_loss', enc_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('test_dec_loss', dec_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('test_char_mismatch', character_mismatch, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('test_accuracy', my_dict['total_accuracy'], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        self.log('test_partial_accuracy', my_dict['partial_accuracy'], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.config['batch_size'], sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.config['learning_rate'], 
            betas=(self.config['beta1'], self.config['beta2']), 
            weight_decay=self.config['weight_decay'], 
            )
        
        if self.config['lr_scheduler'] == 'cosine':
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_steps'], eta_min=self.config['learning_rate']/50)
        elif self.config['lr_scheduler'] == 'cosine_warmup':
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_steps'], eta_min=self.config['learning_rate']/50)
            lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=self.config['num_steps']//10, after_scheduler=lr_scheduler)
        elif self.config['lr_scheduler'] == 'onecycle':
            lr_scheduler = OneCycleLR(optimizer, max_lr=self.config['learning_rate'], total_steps=self.config['num_steps'])
        elif self.config['lr_scheduler'] == 'func':
            lr_scheduler = FuncLR(optimizer, lr_lambda=self._transformer_lr)
        else:
            raise NotImplementedError
        
        scheduler = {"scheduler": lr_scheduler, "interval": "step"}
        
        return [optimizer], scheduler
    
    def _transformer_lr(self, step):
        mult = self.config['n_embd']**-0.5
        step = 1 if step == 0 else step  # Stop div by zero errors
        lr = min(step**-0.5, step * (self.config['warm_up_steps']**-1.5))
        return self.config['learning_rate'] * mult * lr


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    # torch.cuda.empty_cache()
    # gc.collect()

    print("Building tokeniser...")
    tokeniser = util.load_tokeniser('Chemformer/my_vocab.txt', 272)
    print("Finished tokeniser.")

    config['pad_token_id'] = tokeniser.vocab[tokeniser.pad_token]
    config['mask_token_id'] = tokeniser.vocab[tokeniser.mask_token]
    config['mask_ignore_token_ids'] = [tokeniser.begin_token, tokeniser.end_token, tokeniser.pad_token, tokeniser.unk_token, tokeniser.mask_token, tokeniser.sep_token]

    print("Reading dataset...")
    dataset = Zinc('/scratch/arihanth.srikar/zinc/')
    print("Finished dataset.")

    print("Building data module...")
    dm = MoleculeDataModule(
            dataset,
            tokeniser,
            config['batch_size'],
            config['block_size'],
            task='aug',
            val_idxs=dataset.val_idxs,
            test_idxs=dataset.test_idxs,
            train_token_batch_size=None,
            num_buckets=12,
            unified_model=False,
        )
    num_available_cpus = len(os.sched_getaffinity(0))
    num_gpus = torch.cuda.device_count()
    num_workers = num_available_cpus // num_gpus
    dm._num_workers = num_workers
    print(f"Using {str(num_workers)} workers for data module.")
    print("Finished datamodule.")

    dm.setup()
    batches_per_gpu = math.ceil(len(dm.train_dataloader()) / num_gpus)
    train_steps = math.ceil(batches_per_gpu / config['grad_accum']) * config['num_epochs']
    config['num_steps'] = train_steps

    model = Chemformer(config)

    logger = WandbLogger(
        # entity=config['entity'],
        project=config['project'],
        name=config['run'],
        save_dir=config['save_dir'],
        mode='disabled' if not config['log'] else 'online',
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        monitor="val_loss",
        mode="min",
        dirpath=f"{config['save_dir']}/{config['project']}/{config['run']}",
        filename="model-{epoch:02d}-{val_loss:.5f}",
    )

    trainer = pl.Trainer(
        # accelerator='gpu', devices=-1, strategy='ddp_find_unused_parameters_True',
        accelerator='gpu', devices=-1, strategy='auto',
        max_epochs=config['num_epochs'], logger=logger,
        precision='bf16-mixed' if config['set_precision'] else '32-true',
        gradient_clip_val=0.5, gradient_clip_algorithm='norm',
        accumulate_grad_batches=config['grad_accum'],
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        val_check_interval=0.1,
        limit_test_batches=0.1,
    )

    if config['train']:
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)

    else:
        # manually load data
        dm.setup('placeholder')

        device = 'cuda'
        model_ckpt = sorted(glob(f"{config['save_dir']}/{config['project']}/{config['run']}/*.ckpt"))
        model_ckpt = model_ckpt[-1] if len(model_ckpt) > 0 else None
        model = Chemformer.load_from_checkpoint(model_ckpt, config=config) if model_ckpt is not None else Chemformer(config)
        print(f"Loaded model from {model_ckpt}")
        model = model.to(device)

        # also dump the predicted and actual products to a pandas dataframe
        df = pd.DataFrame(columns=['target_smiles', 'predicted_smiles'])
        all_target_smiles = []
        all_predicted_smiles = []

        # generate sequences
        # for (split, split_dm) in [('test', dm.test_dataloader()), ('val', dm.val_dataloader()), ('train', dm.train_dataloader())]:
            # for ft in [0.90, 0.95, 0.99, 0.999]:
        # for (split, split_dm) in [('test', dm.test_dataloader()), ('val', dm.val_dataloader())]:
        for (split, split_dm) in [('test', dm.test_dataloader())]:
            for ft in [0.999]:
                with tqdm(split_dm, desc=f'{split}-filter_threshold={ft}') as pbar:
                    partial_acc = []
                    total_acc   = []
                    for batch in pbar:
                        products  = batch["encoder_input"].transpose(0, 1).to(device)
                        reactants = batch["decoder_input"].transpose(0, 1).to(device)
                        src_mask  = products != model.config['pad_token_id']

                        _, enc, _ = model.encoder(products, mask=src_mask, return_logits_and_embeddings=True)
                        gen_seq = model.decoder.generate(reactants[:, :1], reactants.shape[1], context=enc, context_mask=src_mask, filter_logits_fn=top_k, filter_thres=ft)

                        prods  = [''.join([model.token_decode[t.item()] for t in prod if t.item() != model.config['pad_token_id']])[1:-1] for prod in products]
                        reacts = [''.join([model.token_decode[t.item()] for t in react if t.item() != model.config['pad_token_id']])[1:-1] for react in reactants]
                        gens   = [''.join([model.token_decode[t.item()] for t in gen if t.item() != model.config['pad_token_id']]) for gen in gen_seq]

                        for i, (p, r, g) in enumerate(zip(prods, reacts, gens)):
                            
                            # find first occurence of '&' in g
                            idx = g.find('&')
                            g = g[:idx] if idx != -1 else g
                            # print(f"{i}:{p}\n{r}\n{g}")

                            # add to dataframe
                            all_target_smiles.append(r)
                            all_predicted_smiles.append(g)

                            r_mol = Chem.MolFromSmiles(r)
                            if r_mol is None:
                                continue
                            r_mols = [Chem.MolFromSmiles(r_smi) for r_smi in r.split('.')]
                            g_mol = Chem.MolFromSmiles(g)
                            g_mols = [Chem.MolFromSmiles(g_smi) for g_smi in g.split('.')]

                            acc = 1
                            par_acc = 0
                            for small_g in g_mols:
                                if small_g is not None and r_mol.HasSubstructMatch(small_g) and small_g.HasSubstructMatch(r_mol):
                                    par_acc = 1
                                else:
                                    acc = 0
                            total_acc.append(acc)
                            partial_acc.append(par_acc)

                            pbar.set_postfix({'total_accuray': sum(total_acc)/len(total_acc), 'partial_accuracy': sum(partial_acc)/len(partial_acc)})

        # update dataframe
        df['target_smiles'] = all_target_smiles
        df['predicted_smiles'] = all_predicted_smiles
        # save dataframe
        df.to_csv(f"results/{config['run']}_test.csv", index=False)
            