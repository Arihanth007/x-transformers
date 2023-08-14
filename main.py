import argparse
import torch
from glob import glob

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

parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--num_epochs', type=int, default=4800, help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=6e-4, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1')
parser.add_argument('--beta2', type=float, default=0.95, help='beta2')
parser.add_argument('--weight_decay', type=float, default=1e-1, help='weight decay')

parser.add_argument('--data_dir', type=str, default='data/', help='data directory')
parser.add_argument('--validate_every', type=int, default=500, help='train iterations')
parser.add_argument('--validate_for', type=int, default=100, help='validate iterations')
parser.add_argument('--train', type=bool, default=False, help='whether to train the model')
parser.add_argument('--grad_accum', type=int, default=1, help='gradient accumulation')

parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--is_compile', type=bool, default=False, help='whether to compile the model')
parser.add_argument('--task', type=str, default='uspto50', help='task')
parser.add_argument('--run', type=str, default='exp', help='run name')
parser.add_argument('--project', type=str, default='uspto50', help='project name')
parser.add_argument('--entity', type=str, default='retrosynthesis', help='entity name')
parser.add_argument('--save_dir', type=str, default='', help='save directory')
parser.add_argument('--log', type=bool, default=False, help='whether to log')
parser.add_argument('--set_precision', type=bool, default=False, help='whether to set precision')

config = vars(parser.parse_args())
config['data_dir'] = config["data_dir"] + config["task"]


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    if config['task'] == 'shakespeare_char':
        from trainer.shakespeare import XTModel
        from dataloader.shakespeare import Shakespeare
        train_dataset = Shakespeare(config['data_dir'], 'train', config['block_size'], config['batch_size']*config['validate_every'])
        val_dataset   = Shakespeare(config['data_dir'], 'val', config['block_size'], config['batch_size']*config['validate_for'])
    elif config['task'] == 'uspto50':
        from trainer.uspto import XTModel
        from dataloader.uspto import USPTO50
        train_dataset = USPTO50(config['data_dir'], 'train', config['batch_size']*config['validate_every'])
        val_dataset   = USPTO50(config['data_dir'], 'val', config['batch_size']*config['validate_for'])
        config['vocab_size'] = train_dataset.vocab_size
    else:
        raise NotImplementedError
        
    model = XTModel(config)
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], collate_fn=train_dataset.collate_fn,
        shuffle=True if config['validate_every'] < 0 else False, num_workers=8, pin_memory=True, prefetch_factor=4)
    val_loader   = DataLoader(
        val_dataset, batch_size=config['batch_size'], collate_fn=val_dataset.collate_fn,
        shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=4)

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
        # accelerator='gpu', devices=-1, num_nodes=1, strategy='ddp_find_unused_parameters_True',
        accelerator='gpu', devices=-1, num_nodes=1, strategy='auto',
        max_epochs=-1, logger=logger,
        precision='bf16-mixed' if config['set_precision'] else '32-true',
        gradient_clip_val=0.5, gradient_clip_algorithm='norm',
        accumulate_grad_batches=config['grad_accum'],
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    if config['train']:
        trainer.fit(model, train_loader, val_loader)

    else:
        raise NotImplementedError
