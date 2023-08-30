import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from SmilesPE.pretokenizer import atomwise_tokenizer


class Zinc(Dataset):
    def __init__(self, data_dir: str='/scratch/arihanth.srikar/data/zinc', split: str='train', to_gen: int=-1):
        extra = ''
        
        # dataset files
        # df = pd.read_csv(f'{data_dir}/x001.csv')   # this is 10% of the dataset
        df = pd.read_pickle(f'{data_dir}/zinc.pkl')  # this is the entire dataset
        df = df[df['set'] == split].copy()

        print('Read dataset')
        
        # read entire dataset and convert to list
        self.smiles = df['smiles'].tolist()
        
        # clear memory
        del df
        
        # load specified number of samples
        self.to_gen = to_gen if to_gen > 0 else len(self.smiles)
        
        # token encoder and decoder
        with open(f'{data_dir}/vocab{extra}.txt', 'r') as f:
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
        if self.to_gen != len(self.smiles):
            idx = torch.randint(0, len(self.smiles), (1,)).item()
        
        # treat the smiles as products
        p = self.smiles[idx]
        p = [self.token_encoder[tok] for tok in atomwise_tokenizer(p)]
        
        # append end of products token
        p = [self.token_encoder['<sop>']] + p + [self.token_encoder['<eop>']]
        mask = [1] * len(p)
        
        return torch.tensor(p), torch.tensor(mask)


    def collate_fn(self, batch):
        smiles, mask = zip(*batch)
        smiles = torch.nn.utils.rnn.pad_sequence(smiles, batch_first=True, padding_value=self.token_encoder['<pad>'])
        mask = (smiles != self.token_encoder['<pad>']).bool()
        return smiles, mask