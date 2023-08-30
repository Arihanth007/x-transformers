import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd
    

class GraphDataset(Dataset):
    def __init__(self, data_dir: str='/scratch/arihanth.srikar', split: str='train') -> None:

        # self.data = pd.read_pickle(f'{data_dir}/data/zinc/zinc.pkl')
        data = pd.read_csv(f'{data_dir}/data/zinc/x001.csv')
        data = data[data['split'] == split]
        self.smiles = data['smiles'].tolist()

    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        
        # get graph from smiles
        x = get_graph(self.smiles[idx])
        
        # node features, positions, edge indices, edge features
        node_feats = torch.tensor(x['node_feats'], dtype=torch.int64)  # N*9
        positions  = torch.tensor(x['positions'], dtype=torch.float64) # N*3
        edge_list  = torch.tensor(x['edge_index'], dtype=torch.int64)  # 2*E
        edge_feats = torch.tensor(x['edge_attr'], dtype=torch.int64)   # E*3

        # use 0 index for padding and prepare mask
        node_feats = node_feats + 1 # 0 is reserved for padding
        edge_feats = edge_feats + 1 # 0 is reserved for padding
        mask = torch.ones_like(node_feats).bool()

        # construct adjacency matrix
        row, col = edge_list
        adj_mat = torch.zeros(row.size(0), col.size(0))
        adj_mat[row, col] = 1
        adj_mat[col, row] = 1
        adj_mat[torch.arange(row.size(0)), torch.arange(row.size(0))] = 1

        # contruct N*N*E dense edge features
        dense_edges_feats = torch.zeros((edge_list.size(1), edge_list.size(1), edge_feats.size(1))).int()
        dense_edges_feats[row, col, :] = edge_feats

        return node_feats, positions, mask, adj_mat, dense_edges_feats

    def collate_fn(self, data):

        # unpack the input data
        node_feats, positions, mask, adj_mat, dense_edges_feats = zip(*data)
        
        # fina the largest graph in the batch
        max_nodes = max([feats.size(0) for feats in node_feats])
        
        # pad the adjacency matrix, node features, positions with all 0s
        adj_mat = torch.vstack([F.pad(mat, (0, max_nodes-mat.size(0), 0, max_nodes-mat.size(0)), "constant", 0).unsqueeze(0) for mat in adj_mat])
        node_feats = pad_sequence(node_feats, batch_first=True, padding_value=0)
        positions = pad_sequence(positions, batch_first=True, padding_value=0)
        
        # pad the mask with all False
        mask = pad_sequence(mask, batch_first=True, padding_value=False)
        
        # pad each matrix in dense_edges_feats with all 0s
        dense_edges_feats = torch.vstack([F.pad(mat, (0, 0, 0, max_nodes-mat.size(0), 0, max_nodes-mat.size(0)), "constant", 0).unsqueeze(0) for mat in dense_edges_feats])
        
        return node_feats, positions, mask, adj_mat, dense_edges_feats


def read_smiles(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
    except:
        print("warning: cannot sanitize smiles: ", smile)
        mol = Chem.MolFromSmiles(smile, sanitize=False)
    mol = Chem.AddHs(mol)
    return mol


def rdkit_remove_hs(mol):
    try:
        return Chem.RemoveHs(mol)
    except:
        return Chem.RemoveHs(mol, sanitize=False)
    

def rdkit_mmff(mol):
    try:
        AllChem.MMFFOptimizeMolecule(mol)
        new_mol = rdkit_remove_hs(mol)
        pos = new_mol.GetConformer().GetPositions()
        return new_mol
    except:
        return rdkit_remove_hs(mol)
    

def rdkit_2d_gen(smile):
    m = read_smiles(smile)
    AllChem.Compute2DCoords(m)
    m = rdkit_mmff(m)
    pos = m.GetConformer().GetPositions()
    return m


def rdkit_3d_gen(smile, seed=0):
    mol = read_smiles(smile)
    AllChem.EmbedMolecule(mol, randomSeed=seed, maxAttempts=1000)
    mol = rdkit_mmff(mol)
    pos = mol.GetConformer().GetPositions()
    return mol

allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
        "CHI_SQUAREPLANAR",
        "CHI_OTHER",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
}

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
        safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum())+1,
        allowable_features["possible_chirality_list"].index(str(atom.GetChiralTag())),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(
            allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()
        ),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(
            allowable_features["possible_number_radical_e_list"],
            atom.GetNumRadicalElectrons(),
        ),
        safe_index(
            allowable_features["possible_hybridization_list"],
            str(atom.GetHybridization()),
        ),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
    ]
    
    return atom_feature

def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(
            allowable_features["possible_bond_type_list"], str(bond.GetBondType())
        ),
        allowable_features["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        allowable_features["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
    ]
    
    return bond_feature

def get_graph(smi, include_positions=False):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    # convert SMILES to rdkit molecule object
    mol = rdkit_remove_hs(read_smiles(smi)) if not include_positions else rdkit_3d_gen(smi)

    out = {}
    atom_features_list = []
    positions = []

    for i, atom in enumerate(mol.GetAtoms()):
        atom_features_list.append(atom_to_feature_vector(atom))
        if include_positions:
            pos = mol.GetConformer().GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])

    out["node_feats"] = np.array(atom_features_list, dtype=np.int32)
    out["positions"] = np.array(positions, dtype=np.float32)
    
    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int32).T
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int32)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int32)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int32)

    out["edge_index"] = edge_index
    out["edge_attr"] = edge_attr
    
    return out

def read_data(file_name: str='../data/ld50_train.sdf'):
    very_toxic = []
    nontoxic = []
    ld50_vals = []
    smiles = []
    mols = []

    inf = open(file_name,'rb')
    with Chem.ForwardSDMolSupplier(inf) as suppl:
        for mol in suppl:
            if mol is not None:
                vt = mol.GetProp('very_toxic')
                nt = mol.GetProp('nontoxic')
                ld50 = mol.GetProp('LD50_mgkg')
                smi = mol.GetProp('Canonical_QSARr')

                very_toxic.append(vt)
                nontoxic.append(nt)
                ld50_vals.append(ld50)
                smiles.append(smi)
                mols.append(mol)

    return very_toxic, nontoxic, ld50_vals, smiles, mols