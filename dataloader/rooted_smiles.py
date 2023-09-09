from itertools import chain

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class RootedSmilesDataset(Dataset):
    def __init__(self, data_dir: str='/scratch/arihanth.srikar', split: str='train', to_gen: int=-1, extra: str='') -> None:

        # source or input is the product
        with open(f'{data_dir}/{split}/src-{split}.txt', 'r') as f:
            self.products = [line.split() for line in f.read().splitlines()]
        
        # target is the reactant
        with open(f'{data_dir}/{split}/tgt-{split}.txt', 'r') as f:
            self.reactants = [line.split() for line in f.read().splitlines()]

        # token encoder and decoder
        with open(f'{data_dir}/vocab{extra}.txt', 'r') as f:
            self.token_decoder = f.read().splitlines()
        self.token_encoder = {k: v for v, k in enumerate(self.token_decoder)}

        # sanity check the tokenizer
        print(f'Performing sanity check on vocab and tokenizer...')
        reactant_set = set(chain.from_iterable(self.reactants))
        product_set = set(chain.from_iterable(self.products))
        all_chars = reactant_set.union(product_set)
        assert all_chars <= set(self.token_encoder.keys()), "Tokenizer is not consistent with the dataset"

        self.vocab_size = len(self.token_decoder)
        self.pad_token_id = self.token_encoder['<pad>']
        self.mask_token_id = self.token_encoder['<mask>']
        self.mask_ignore_token_ids = [v for k, v in self.token_encoder.items() if '<' in k and '>' in k]

        self.to_gen = to_gen if to_gen > 0 else len(self.reactants)

    def __len__(self):
        return self.to_gen
    
    def __getitem__(self, idx):

        # pick random indices if not utilizing entire dataset
        to_gen = len(self.reactants)
        if self.to_gen != to_gen:
            idx = torch.randint(0, to_gen, (1,)).item()
        
        # treat the docoder part as reactants
        # r = ''.join(self.reactants[idx])
        # r = [self.token_encoder[tok] for tok in atomwise_tokenizer(r)]
        r = [self.token_encoder[tok] for tok in self.reactants[idx]]
        p = [self.token_encoder[tok] for tok in self.products[idx]]
        
        # append end of products token
        r = [self.token_encoder['<sos>']] + r + [self.token_encoder['<eos>']]
        p = [self.token_encoder['<sos>']] + p + [self.token_encoder['<eos>']]
        src_mask = [True] * len(p)
        dec_mask = [True] * len(r)

        # convert to tensors
        r, dec_mask = torch.tensor(r), torch.tensor(dec_mask).bool()
        p, src_mask = torch.tensor(p), torch.tensor(src_mask).bool()

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
