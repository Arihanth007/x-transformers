import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence



class Shakespeare(Dataset):
    def __init__(self, data_dir: str='data/shakespeare_char', split: str='train', block_size: int=1024, num_samples: int=2000) -> None:
        self.split = split
        self.data = np.memmap(f'{data_dir}/{split}.bin', dtype=np.uint16, mode='r')
        self.sz = len(self.data) - block_size
        self.num_samples = num_samples
        self.block_size = block_size

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, i):
        i = torch.randint(0, self.sz, (1,)).item()
        src = torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64)).long()
        tgt = torch.from_numpy(self.data[i+self.block_size:i+2*self.block_size].astype(np.int64)).long()
        src_mask = torch.ones_like(src).bool()
        return src, tgt, src_mask
    
    def collate_fn(self, batch):
        src, tgt, src_mask = zip(*batch)
        src = pad_sequence(src, batch_first=True, padding_value=0)
        tgt = pad_sequence(tgt, batch_first=True, padding_value=0)
        src_mask = pad_sequence(src_mask, batch_first=True, padding_value=False)
        return src, tgt, src_mask