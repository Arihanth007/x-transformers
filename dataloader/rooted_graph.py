import pandas as pd
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from data.zinc.preprocess import get_graph
from SmilesPE.pretokenizer import atomwise_tokenizer


class RootedGraphDataset(Dataset):
    def __init__(self, data_dir: str='/scratch/arihanth.srikar', split: str='train', to_gen: int=-1) -> None:
        extra = ''

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
        
        # get graph from smiles
        x = get_graph(''.join(self.products[idx]))
        
        # node features, positions, edge indices, edge features
        node_feats = torch.tensor(x['node_feats'], dtype=torch.int64)  # N*9
        positions  = torch.tensor(x['positions'], dtype=torch.float64) # N*3
        edge_list  = torch.tensor(x['edge_index'], dtype=torch.int64)  # 2*E
        edge_feats = torch.tensor(x['edge_attr'], dtype=torch.int64)   # E*3

        # use 0 index for padding and prepare src_mask
        node_feats = node_feats + 1 # 0 is reserved for padding
        edge_feats = edge_feats + 1 # 0 is reserved for padding
        src_mask = torch.ones(node_feats.size(0)).bool()

        # construct adjacency matrix
        row, col = edge_list
        adj_mat = torch.zeros(row.size(0), col.size(0))
        adj_mat[row, col] = 1
        adj_mat[col, row] = 1
        adj_mat[torch.arange(row.size(0)), torch.arange(row.size(0))] = 1

        # contruct N*N*E dense edge features
        dense_edges_feats = torch.zeros((edge_list.size(1), edge_list.size(1), edge_feats.size(1)), dtype=torch.int64)
        dense_edges_feats[row, col, :] = edge_feats

        # treat the docoder part as reactants
        # r = ''.join(self.reactants[idx])
        # r = [self.token_encoder[tok] for tok in atomwise_tokenizer(r)]
        r = [self.token_encoder[tok] for tok in self.reactants[idx]]
        
        # append end of products token
        r = [self.token_encoder['<sos>']] + r + [self.token_encoder['<eos>']]
        dec_mask = [True] * len(r)

        # convert to tensors
        r, dec_mask = torch.tensor(r), torch.tensor(dec_mask).bool()

        return node_feats, positions, src_mask, adj_mat, dense_edges_feats, r, dec_mask

    def collate_fn(self, data):

        # unpack the input data
        node_feats, positions, src_mask, adj_mat, dense_edges_feats, r, dec_mask = zip(*data)
        
        # find the largest graph in the batch
        max_nodes = max([feats.size(0) for feats in node_feats])
        
        # pad the adjacency matrix, node features, positions with all 0s
        adj_mat = torch.vstack([F.pad(mat, (0, max_nodes-mat.size(0), 0, max_nodes-mat.size(0)), "constant", 0).unsqueeze(0) for mat in adj_mat])
        node_feats = pad_sequence(node_feats, batch_first=True, padding_value=0)
        positions = pad_sequence(positions, batch_first=True, padding_value=0)

        # pad the decoder stuff
        r = pad_sequence(r, batch_first=True, padding_value=self.pad_token_id)
        dec_mask = pad_sequence(dec_mask, batch_first=True, padding_value=False).bool()
        
        # pad the src_mask with all False
        src_mask = pad_sequence(src_mask, batch_first=True, padding_value=False).bool()
        
        # pad each matrix in dense_edges_feats with all 0s
        dense_edges_feats = torch.vstack([F.pad(mat, (0, 0, 0, max_nodes-mat.size(0), 0, max_nodes-mat.size(0)), "constant", 0).unsqueeze(0) for mat in dense_edges_feats])
        
        return node_feats, positions, src_mask, adj_mat, dense_edges_feats, r, dec_mask