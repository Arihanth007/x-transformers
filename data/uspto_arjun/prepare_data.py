import pandas as pd
from tqdm import tqdm

with open('vocab_8192.txt', 'r') as f:
    idx2token = f.read().splitlines()
token2idx = {k: v for v, k in enumerate(idx2token)}

def read_vocabulary(file_path):
    with open(file_path, 'r') as file:
        vocabulary = set(line.strip() for line in file)
    return vocabulary

vocabulary_file_path = 'vocab_8192.txt'

vocabulary = read_vocabulary(vocabulary_file_path)

def tokenize_sentence(input_sentence):
    tokens = []
    i = 0
    while i < len(input_sentence):
        found = False
        for j in range(len(vocabulary), 0, -1):
            token = input_sentence[i:i+j]
            if token in vocabulary:
                tokens.append(token)
                i += j
                found = True
                break
        if not found:
            tokens.append(input_sentence[i])
            i += 1
    return tokens

# Example input sentence
# input_sentence = 'Cc1cc(N)c2cn[nH]c2c1.Fc1ccc(I)c(F)c1'

# Tokenize the input sentence
# tokenized_tokens = tokenize_sentence(input_sentence)

# Print the tokenized tokens
# print(tokenized_tokens)

df = pd.read_pickle('processed.pickle')
reactants = df['reactants_mol'].apply(lambda x: '.'.join(x))
products  = df['products_mol'].apply(lambda x: '.'.join(x))
r_tokens = []
p_tokens = []

for r, p in tqdm(zip(reactants, products), total=len(reactants)):
    # for j in range(20):
    #     print('_', end='')
    # print()
    # print(f'Smiles String: {r}')
    r_toks = tokenize_sentence(r)
    # print()
    # print(f'Tokenised Smiles String: {r_toks}')
    # print()
    # for j in range(20):
    #     print('_', end='')
    p_toks = tokenize_sentence(p)
    r_idxs = [token2idx[tok] for tok in r_toks]
    p_idxs = [token2idx[tok] for tok in p_toks]
    # r_idxs = [token2idx['<sos>']] + r_idxs + [token2idx['<eos>']]
    # p_idxs = [token2idx['<sos>']] + p_idxs + [token2idx['<eos>']]
    r_tokens.append(r_idxs)
    p_tokens.append(p_idxs)

df['reactants_tokens'] = r_tokens
df['products_tokens'] = p_tokens
pd.to_pickle(df, 'processed_tokens_8192.pickle')

# print("done!!")
