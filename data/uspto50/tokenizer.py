import pandas as pd
from SmilesPE.pretokenizer import atomwise_tokenizer


df = pd.read_pickle('processed.pickle')
print(df.head())

vocab = dict()
for entry in df['reactants_mol']:
    smi = '.'.join(entry)
    toks = atomwise_tokenizer(smi)
    for tok in toks:
        if tok in vocab:
            vocab[tok] += 1
        else:
            vocab[tok] = 1
print(f'vocab size: {len(vocab)}')

for entry in df['products_mol']:
    smi = '.'.join(entry)
    toks = atomwise_tokenizer(smi)
    for tok in toks:
        if tok in vocab:
            vocab[tok] += 1
        else:
            vocab[tok] = 1
print(f'vocab size: {len(vocab)}')

with open('vocab.txt', 'w') as f:
    for k, v in vocab.items():
        f.write(f'{k}\n')
    f.write('<unk>\n')
    f.write('<sos>\n')
    f.write('<eos>\n')
    f.write('<mask>\n')
    f.write('<sum_pred>\n')
    f.write('<sum_react>\n')
    f.write('<0>\n')
    f.write('<1>\n')
    f.write('<2>\n')
    f.write('<3>\n')
    f.write('<pad>\n')
