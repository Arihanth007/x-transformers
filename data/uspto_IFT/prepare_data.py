from tqdm import tqdm
import pandas as pd
import numpy as np

import codecs
from SmilesPE.pretokenizer import atomwise_tokenizer
from SmilesPE.tokenizer import *

#Supress warnings from RDKit
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

import sys
sys.path.append('../../')
from Levy.levenshteinaugment.levenshtein import Levenshtein_augment

def augment(reactants, products, num_IFT=-1):
    reactants = sorted(reactants, key=lambda x: len(x), reverse=True)
    products  = sorted(products, key=lambda x: len(x), reverse=True)
    
    new_reactants, new_products, all_score = [], [], []
    for i in range(1 if num_IFT == -1 else len(reactants), len(reactants)+1):
        reactant, product = '.'.join(reactants[:i]), '.'.join(products[:i])
    
        pairs = augmenter.levenshtein_pairing(reactant, product)
        augmentations = augmenter.sample_pairs(pairs)
    
        for new_reactant, new_product, score in augmentations:
            new_reactants.append(new_reactant)
            new_products.append(new_product)
            all_score.append(score)
    
    return new_reactants, new_products, all_score


augmenter = Levenshtein_augment(source_augmentation=1, randomization_tries=1000)

main_df = pd.read_pickle('processed.pickle')

train = main_df.sample(frac=0.8, random_state=200)
test  = main_df.drop(train.index)

val  = test.sample(frac=0.5, random_state=200)
test = test.drop(val.index)

# for df, name in zip([val, test, train], ['val', 'test', 'train']):
for df, name in zip([test], ['test']):
    rf = open(f'{name}/reactants.txt', 'w')
    pf = open(f'{name}/products.txt', 'w')
    for i, (reactants, products) in enumerate(tqdm(zip(df['reactants_mol'], df['products_mol']), total=len(df), desc=name)):
        new_reactants, new_products, score = augment(reactants, products, num_IFT=-1 if name != 'test' else 1)
        # print(f'{".".join(reactants)} -> {".".join(products)}')
        for reactant, product, sc in zip(new_reactants, new_products, score):
            # print(f'{sc:.2f}: {reactant} -> {product}')
            reactant = ' '.join(atomwise_tokenizer(reactant))
            product  = ' '.join(atomwise_tokenizer(product))
            rf.write(f'{reactant}\n')
            pf.write(f'{product}\n')
        # print()
    rf.close()
    pf.close()
