from utils import load_csv
import pickle
import numpy as np

parent_folder = './dataset1/'

X, y, vocab = load_csv(parent_folder, max_seq_len=500)
print("X shape:", X.shape)
print("y shape:", y.shape)


np.save(f'{parent_folder}X.npy', X)
np.save(f'{parent_folder}y.npy', y)

with open(f'{parent_folder}vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)