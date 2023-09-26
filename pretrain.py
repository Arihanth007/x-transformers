import gc
import argparse
from glob import glob
import torch
import pandas as pd

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


parser = argparse.ArgumentParser(description='Retrosynthesis')

parser.add_argument('--block_size', type=int, default=1024, help='block size')
parser.add_argument('--vocab_size', type=int, default=256, help='vocab size')
parser.add_argument('--n_layer', type=int, default=12, help='number of layers')
parser.add_argument('--n_head', type=int, default=12, help='number of heads')
parser.add_argument('--n_embd', type=int, default=768, help='embedding dimension')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
parser.add_argument('--bias', type=bool, default=False, help='whether to use bias in attention layer')
parser.add_argument('--use_pos_emb', type=bool, default=False, help='whether to use positional embeddings')
parser.add_argument('--use_rotary_emb', type=bool, default=False, help='whether to use rotary embeddings')
parser.add_argument('--use_rel_pos_emb', type=bool, default=False, help='whether to use relative positional embeddings')
parser.add_argument('--mask_prob', type=float, default=0.0, help='mask probability')

parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_epochs', type=int, default=4800, help='number of epochs')
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
parser.add_argument('--grad_accum', type=int, default=1, help='gradient accumulation')

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


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    torch.cuda.empty_cache()
    gc.collect()
    
    # pretraining on zinc
    if config['task'] == 'zinc':
        config['data_dir'] = '/scratch/arihanth.srikar/data/zinc'
        # df = pd.read_csv(f'/scratch/arihanth.srikar/x001.csv')     # this is 10% of the dataset
        df = pd.read_pickle(f'{config["data_dir"]}/zinc.pkl')   # this is the entire dataset
        from trainer.zinc_pretrain import XTModel
        from dataloader.zinc import Zinc
        train_dataset = Zinc(config['data_dir'], 'train', config['batch_size']*config['validate_every'], df)
        val_dataset   = Zinc(config['data_dir'], 'val', config['batch_size']*config['validate_for'], df)
        test_dataset  = Zinc(config['data_dir'], 'val', config['batch_size']*config['generate_for'], df)
        config['pad_token_id'] = train_dataset.pad_token_id
        config['mask_token_id'] = train_dataset.mask_token_id
        config['mask_ignore_token_ids'] = train_dataset.mask_ignore_token_ids
        print(len(config['mask_ignore_token_ids']), config['mask_ignore_token_ids'])

    # finetune on rooted uspto50k
    elif 'rooted_smiles' in config['task']:
        config['data_dir'] = 'data/rooted'
        from trainer.uspto_finetune import XTModel
        from dataloader.rooted_smiles import RootedSmilesDataset
        train_dataset = RootedSmilesDataset(config['data_dir'], 'train', config['batch_size']*config['validate_every'], config['vocab_file'])
        val_dataset   = RootedSmilesDataset(config['data_dir'], 'val', config['batch_size']*config['validate_for'], config['vocab_file'])
        test_dataset  = RootedSmilesDataset(config['data_dir'], 'test', config['batch_size']*config['validate_for'], config['vocab_file'])
        # config['vocab_size'] = train_dataset.vocab_size
        config['pad_token_id'] = train_dataset.pad_token_id
        config['mask_token_id'] = train_dataset.mask_token_id
        config['mask_ignore_token_ids'] = train_dataset.mask_ignore_token_ids
    
    # finetune on uspto50k
    elif config['task'] == 'finetune_smiles':
        config['data_dir'] = 'data/uspto50'
        from trainer.uspto_finetune import XTModel
        from dataloader.uspto import USPTO50
        train_dataset = USPTO50(config['data_dir'], 'train', config['batch_size']*config['validate_every'], config['vocab_file'])
        val_dataset   = USPTO50(config['data_dir'], 'val', config['batch_size']*config['validate_for'], config['vocab_file'])
        test_dataset  = USPTO50(config['data_dir'], 'test', config['batch_size']*config['validate_for'], config['vocab_file'])
        # config['vocab_size'] = train_dataset.vocab_size
        config['pad_token_id'] = train_dataset.pad_token_id
        config['mask_token_id'] = train_dataset.mask_token_id
        config['mask_ignore_token_ids'] = train_dataset.mask_ignore_token_ids
    
    # finetune on levy uspto50k IFT
    elif config['task'] == 'finetune_levy_ift':
        config['data_dir'] = 'data/uspto_IFT'
        from trainer.uspto_finetune import XTModel
        from dataloader.levy_smiles_ift import USPTOIFT
        train_dataset = USPTOIFT(config['data_dir'], 'train', config['batch_size']*config['validate_every'])
        val_dataset   = USPTOIFT(config['data_dir'], 'val', config['batch_size']*config['validate_for'])
        test_dataset  = USPTOIFT(config['data_dir'], 'test', config['batch_size']*config['validate_for'])
        # config['vocab_size'] = train_dataset.vocab_size
        config['pad_token_id'] = train_dataset.pad_token_id
        config['mask_token_id'] = train_dataset.mask_token_id
        config['mask_ignore_token_ids'] = train_dataset.mask_ignore_token_ids
    
    else:
        raise NotImplementedError
        
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], collate_fn=train_dataset.collate_fn,
        shuffle=True if config['validate_every'] < 0 else False, num_workers=16)
    val_loader   = DataLoader(
        val_dataset, batch_size=config['batch_size'], collate_fn=val_dataset.collate_fn,
        shuffle=False, num_workers=16)
    test_loader  = DataLoader(
        test_dataset, batch_size=config['batch_size'], collate_fn=val_dataset.collate_fn,
        shuffle=False, num_workers=16)
    
    # train batches
    config['num_batches'] = len(train_loader) * config['num_epochs']
    
    model = XTModel(config, train_dataset.token_encoder, train_dataset.token_decoder) if 'finetune' in config['task'] else XTModel(config, task=config['sub_task'])

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
        monitor="val_loss" if config['sub_task'] == 'enc' else "val_char_mismatch",
        mode="min",
        dirpath=f"{config['save_dir']}/{config['project']}/{config['run']}",
        filename="model-{epoch:02d}-{val_loss:.5f}" if config['sub_task'] == 'enc' else "model-{epoch:02d}-{val_char_mismatch:.5f}",
    )

    trainer = pl.Trainer(
        accelerator='gpu', devices=config['device_ids'], strategy='ddp_find_unused_parameters_True',
        max_epochs=-1, logger=logger,
        precision='bf16-mixed' if config['set_precision'] else '32-true',
        gradient_clip_val=0.5, gradient_clip_algorithm='norm',
        accumulate_grad_batches=config['grad_accum'],
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    if config["finetune"]:
        # model_ckpt = sorted(glob(f"{config['save_dir']}/{config['project']}/{config['run']}/*.ckpt"))[0]
        model_ckpt = sorted(glob(f"/scratch/arihanth.srikar/uspto50/pretrain-zinc-decode/*.ckpt"))[0]
        # model_ckpt = sorted(glob(f"/scratch/arihanth.srikar/uspto50/pretrain-zinc-enc/*.ckpt"))[0]

        if config['train']:
            model = XTModel.load_from_checkpoint(model_ckpt, config=config, token_encoder=train_dataset.token_encoder, token_decoder=train_dataset.token_decoder)
            trainer.fit(model, train_loader, val_loader)
        else:
            model_ckpt = sorted(glob(f"/scratch/arihanth.srikar/uspto50/train_scratch_levy_ift/*.ckpt"))[-1]
            trainer.validate(model, val_loader, ckpt_path=model_ckpt)
            trainer.test(model, test_loader, ckpt_path=model_ckpt)
    
    else:

        if config['train']:
            model_ckpt = sorted(glob(f"/scratch/arihanth.srikar/uspto50/pretrain-zinc-enc/*.ckpt"))
            model_ckpt = model_ckpt[-1] if len(model_ckpt) else None
            print('Training from scratch' if model_ckpt is None else f'Resuming from {model_ckpt}')

            trainer.fit(model, train_loader, val_loader, ckpt_path=model_ckpt)

        else:
            model_ckpt = sorted(glob(f"{config['save_dir']}/{config['project']}/{config['run']}/*.ckpt"))
            model_ckpt = model_ckpt[0] if len(model_ckpt) else None
            print('Randomly initializing model' if model_ckpt is None else f'Loading {model_ckpt}')

            trainer.test(model, test_loader)
