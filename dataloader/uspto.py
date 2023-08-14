import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class USPTO50(Dataset):
    def __init__(self, data_dir: str='data/uspto50', split: str='train', to_gen: int=-1):
        
        # dataset files
        df = pd.read_pickle(f'{data_dir}/processed_tokens.pickle')
        indices = np.load(f'{data_dir}/{split}_indices.npy')
        
        # read entire dataset and convert to list
        self.reactants = df['reactants_tokens'].tolist()
        self.products = df['products_tokens'].tolist()
        
        # split based on saved indices
        self.reactants = [self.reactants[i] for i in indices]
        self.products = [self.products[i] for i in indices]
        
        # load specified number of samples
        self.to_gen = to_gen if to_gen > 0 else len(self.reactants)
        
        # token encoder and decoder
        with open(f'{data_dir}/vocab.txt', 'r') as f:
            self.token_decoder = f.read().splitlines()
        self.token_encoder = {k: v for v, k in enumerate(self.token_decoder)}

        self.vocab_size = len(self.token_decoder)

    def __len__(self):
        return self.to_gen

    def __getitem__(self, idx):
        
        # pick random indices if not utilizing entire dataset
        if self.to_gen != len(self.reactants):
            idx = torch.randint(0, len(self.reactants), (1,)).item()
        
        # reactants -> output of model
        # products  -> input to model
        r, p = self.reactants[idx], self.products[idx]
        
        # prepend start of reactants token
        r = [self.token_encoder['<sor>']] + r + [self.token_encoder['<eor>']]
        
        # append end of products token
        p = [self.token_encoder['<sop>']] + p + [self.token_encoder['<eop>']]
        
        return torch.tensor(r), torch.tensor(p)


    def collate_fn(self, batch):
        reactants, products = zip(*batch)
        reactants = torch.nn.utils.rnn.pad_sequence(reactants, batch_first=True, padding_value=self.token_encoder['<pad>'])
        products = torch.nn.utils.rnn.pad_sequence(products, batch_first=True, padding_value=self.token_encoder['<pad>'])
        return reactants, products