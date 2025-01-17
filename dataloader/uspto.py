import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class USPTO50(Dataset):
    def __init__(self, data_dir: str='data/uspto50', split: str='train', to_gen: int=-1, vocab_file: str='') -> None:
        # data_dir = 'data/uspto_arjun'
        extra = ''
        
        # dataset files
        df = pd.read_pickle(f'{data_dir}/processed_tokens{extra}.pickle')
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
        vocab_file = vocab_file if vocab_file != '' else f'{data_dir}/vocab{extra}.txt'
        with open(vocab_file, 'r') as f:
            self.token_decoder = f.read().splitlines()
        self.token_encoder = {k: v for v, k in enumerate(self.token_decoder)}

        self.vocab_size = len(self.token_decoder)
        self.pad_token_id = self.token_encoder['<pad>']
        self.mask_token_id = self.token_encoder['<mask>']
        self.mask_ignore_token_ids = [v for k, v in self.token_encoder.items() if '<' in k and '>' in k]

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
        r = [self.token_encoder['<sos>']] + r + [self.token_encoder['<eos>']]
        
        # append end of products token
        p = [self.token_encoder['<sos>']] + p + [self.token_encoder['<eos>']]
        mask = [1] * len(p)
        
        return torch.tensor(r), torch.tensor(p), torch.tensor(mask)


    def collate_fn(self, batch):
        reactants, products, mask = zip(*batch)
        reactants = torch.nn.utils.rnn.pad_sequence(reactants, batch_first=True, padding_value=self.token_encoder['<pad>'])
        products = torch.nn.utils.rnn.pad_sequence(products, batch_first=True, padding_value=self.token_encoder['<pad>'])
        mask = (products != self.token_encoder['<pad>']).bool()
        return reactants, products, mask