from itertools import chain

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class USPTOIFT(Dataset):
    
    def __init__(self, data_dir: str='data/uspto_IFT', split: str='val', to_gen: int=-1, extra: str='') -> None:

        # target is the reactant
        with open(f'{data_dir}/{split}/reactants.txt', 'r') as f:
            self.reactants = f.read().splitlines()
        self.reactants = [r.split(' ') for r in self.reactants]

        # source or input is the product
        with open(f'{data_dir}/{split}/products.txt', 'r') as f:
            self.products = f.read().splitlines()
        self.products = [p.split(' ') for p in self.products]

        # verify that the dataset is consistent
        assert len(self.reactants) == len(self.products), 'Mismatched length of reactants and products'
        self.to_gen = to_gen if to_gen > 0 else len(self.reactants)

        # vocab and tokenizer
        with open(f'{data_dir}/vocab{extra}.txt', 'r') as f:
            self.token_decoder = f.read().splitlines()
        self.token_encoder = {t: i for i, t in enumerate(self.token_decoder)}

        # sanity check the tokenizer
        print(f'Performing sanity check on vocab and tokenizer...')
        reactant_set = set(chain.from_iterable(self.reactants))
        product_set = set(chain.from_iterable(self.products))
        all_chars = reactant_set.union(product_set)
        assert all_chars <= set(self.token_encoder.keys()), "Tokenizer is not consistent with the dataset"

        # additional information
        self.vocab_size = len(self.token_decoder)
        self.pad_token_id = self.token_encoder['<pad>']
        self.mask_token_id = self.token_encoder['<mask>']
        self.mask_ignore_token_ids = [v for k, v in self.token_encoder.items() if '<' in k and '>' in k]

    def __len__(self):
        return self.to_gen
    
    def __getitem__(self, idx):
        r, p = self.reactants[idx], self.products[idx]
        num_reactants, num_products = r.count('.')+1, p.count('.')+1

        # avoid 4 reactants
        while num_reactants > 3:
            idx = torch.randint(0, len(self.reactants), (1,)).item()
            r, p = self.reactants[idx], self.products[idx]
            num_reactants, num_products = r.count('.')+1, p.count('.')+1

        r = [f'<{num_reactants}>'] + ['<sos>'] + r + ['<eos>']
        p = [f'<{num_products}>']  + ['<sos>'] + p + ['<eos>']
        
        r = [self.token_encoder[t] for t in r]
        p = [self.token_encoder[t] for t in p]

        src_mask = [True] * len(p)

        r, p, src_mask = torch.tensor(r), torch.tensor(p), torch.tensor(src_mask).bool()

        return r, p, src_mask
    
    def collate_fn(self, data):

        # unpack the input data
        r, p, src_mask = zip(*data)
        
        # pad the encoder stuff
        p = pad_sequence(p, batch_first=True, padding_value=self.pad_token_id)
        src_mask = pad_sequence(src_mask, batch_first=True, padding_value=False).bool()
        
        # pad the decoder stuff
        r = pad_sequence(r, batch_first=True, padding_value=self.pad_token_id)
        
        return r, p, src_mask