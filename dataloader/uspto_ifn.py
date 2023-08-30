import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class USPTO50(Dataset):
    def __init__(self, data_dir: str='data/uspto50', split: str='train', to_gen: int=-1):
        
        # dataset files
        df = pd.read_pickle(f'{data_dir}/final_data.pickle')

        # take the correct split
        df = df[df['set'] == split]
        
        # tokenised molecules
        self.reactant_token = df['reactant_token'].to_list()
        self.product_token  = df['product_token'].to_list()

        # positions embeddings w.r.t bfs and dfs
        self.reactant_bfs_pos = df['reactant_bfs_depth'].to_list()
        self.reactant_dfs_pos = df['reactant_dfs_depth'].to_list()
        self.product_bfs_pos  = df['product_bfs_depth'].to_list()
        self.product_dfs_pos  = df['product_dfs_depth'].to_list()

        # each molecule has its own ID
        self.mol_id = [i for token_list in self.reactant_token for i, _ in enumerate(token_list)]

        # linearise the lists
        num_reactans = [len(r) for r in self.reactant_token]
        self.reactant_token = [sub_entry for entry in self.reactant_token for sub_entry in entry]
        self.product_token  = [sub_entry for i, entry in enumerate(self.product_token) for sub_entry in entry for _ in range(num_reactans[i])]
        self.reactant_bfs_pos  = [sub_entry for entry in self.reactant_bfs_pos for sub_entry in entry]
        self.reactant_dfs_pos  = [sub_entry for entry in self.reactant_dfs_pos for sub_entry in entry]
        self.product_bfs_pos   = [sub_entry for i, entry in enumerate(self.product_bfs_pos) for sub_entry in entry for _ in range(num_reactans[i])]
        self.product_dfs_pos   = [sub_entry for i, entry in enumerate(self.product_dfs_pos) for sub_entry in entry for _ in range(num_reactans[i])]

        # load specified number of samples
        total_data_points = len(self.reactant_token)
        self.to_gen = to_gen if to_gen > 0 else total_data_points
        
        # token encoder and decoder
        with open(f'{data_dir}/vocab.txt', 'r') as f:
            self.token_decoder = f.read().splitlines()
        self.token_encoder = {k: v for v, k in enumerate(self.token_decoder)}

        self.vocab_size = len(self.token_decoder)
        self.pad_token_id = self.token_encoder['<pad>']

    def __len__(self):
        return self.to_gen

    def __getitem__(self, idx):
        
        # pick random indices if not utilizing entire dataset
        if self.to_gen != len(self.reactant_token):
            idx = torch.randint(0, len(self.reactant_token), (1,)).item()
        
        # reactants -> output of model
        # products  -> input to model
        reactants, products = self.reactant_token[idx], self.product_token[idx]
        
        # positions
        reactant_bfs_pos, reactant_dfs_pos = self.reactant_bfs_pos[idx], self.reactant_dfs_pos[idx]
        product_bfs_pos, product_dfs_pos = self.product_bfs_pos[idx], self.product_dfs_pos[idx]

        # offset postions to accomodate special tokes
        reactant_bfs_pos = [i+1 for i in reactant_bfs_pos]
        reactant_dfs_pos = [i+1 for i in reactant_dfs_pos]
        product_bfs_pos  = [i+1 for i in product_bfs_pos]
        product_dfs_pos  = [i+1 for i in product_dfs_pos]
        reactant_bfs_pos = [0] + reactant_bfs_pos + [self.token_encoder['<mask>']]
        reactant_dfs_pos = [0] + reactant_dfs_pos + [self.token_encoder['<mask>']]
        product_bfs_pos  = [0] + product_bfs_pos + [self.token_encoder['<mask>']]
        product_dfs_pos  = [0] + product_dfs_pos + [self.token_encoder['<mask>']]

        # mol ID
        mol_id = self.mol_id[idx]
        
        # prepend start of reactants token
        reactants = [f'<{mol_id}>'] + reactants + ['<eor>']
        reactants = [self.token_encoder[token] for token in reactants]
        
        # append end of products token
        products = ['<sop>'] + products + ['<eop>']
        products = [self.token_encoder[token] for token in products]
        mask = [1] * len(products)

        reactants, products = torch.tensor(reactants), torch.tensor(products)
        reactant_bfs_pos, reactant_dfs_pos = torch.tensor(reactant_bfs_pos), torch.tensor(reactant_dfs_pos)
        product_bfs_pos, product_dfs_pos = torch.tensor(product_bfs_pos), torch.tensor(product_dfs_pos)
        mol_id, mask = torch.tensor(mol_id), torch.tensor(mask)
        
        return reactants, products, mask, reactant_bfs_pos, reactant_dfs_pos, product_bfs_pos, product_dfs_pos, mol_id


    def collate_fn(self, batch):
        reactants, products, mask, reactant_bfs_pos, reactant_dfs_pos, product_bfs_pos, product_dfs_pos, mol_id = zip(*batch)
        reactants = torch.nn.utils.rnn.pad_sequence(reactants, batch_first=True, padding_value=self.token_encoder['<pad>'])
        products = torch.nn.utils.rnn.pad_sequence(products, batch_first=True, padding_value=self.token_encoder['<pad>'])
        mask = (products != self.token_encoder['<pad>']).bool()
        reactant_bfs_pos = torch.nn.utils.rnn.pad_sequence(reactant_bfs_pos, batch_first=True, padding_value=self.token_encoder['<mask>'])
        reactant_dfs_pos = torch.nn.utils.rnn.pad_sequence(reactant_dfs_pos, batch_first=True, padding_value=self.token_encoder['<mask>'])
        product_bfs_pos  = torch.nn.utils.rnn.pad_sequence(product_bfs_pos, batch_first=True, padding_value=self.token_encoder['<mask>'])
        product_dfs_pos  = torch.nn.utils.rnn.pad_sequence(product_dfs_pos, batch_first=True, padding_value=self.token_encoder['<mask>'])
        mol_id = torch.vstack(mol_id)
        return reactants, products, mask, reactant_bfs_pos, reactant_dfs_pos, product_bfs_pos, product_dfs_pos, mol_id