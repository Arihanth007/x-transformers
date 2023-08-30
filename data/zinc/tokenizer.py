from collections import Counter
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from SmilesPE.pretokenizer import atomwise_tokenizer


vocab = Counter()
df = pd.read_pickle(f'../uspto50/final_data.pickle')
for smi in tqdm(df['reactants_mol'].tolist() + df['products_mol'].tolist()):
    toks = atomwise_tokenizer('.'.join(smi))
    vocab.update(Counter(toks))
print(f'vocab size at end of USPTO-50: {len(vocab)}')

for i in range(1, 10):
    df = pd.read_csv(f'/scratch/arihanth.srikar/x00{i}.csv')

    for smi in tqdm(df['smiles']):
        toks = atomwise_tokenizer(smi)
        vocab.update(Counter(toks))
    print(f'vocab size at end of {i}: {len(vocab)}')

# histogram
plt.hist(list(vocab.values()), bins=100)
plt.savefig('hist.png')

with open('new_vocab.txt', 'w') as f:
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
