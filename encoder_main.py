import pandas as pd
import numpy as np
import os
import torch
import pickle
# import torchtext; torchtext.disable_torchtext_deprecation_warning()
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils import *

parent_folder = './dataset1/'
X_path = os.path.join(parent_folder, 'X.npy')
y_path = os.path.join(parent_folder, 'y.npy')
vacob_path = os.path.join(parent_folder, 'vocab.pkl')

embedding_dim = 300
hidden_dim = 256
output_dim = 1
n_epochs = 10
nhead = 10
num_encoder_layers = 3
lr=0.001

X, y = torch.LongTensor(np.load(X_path)), torch.LongTensor(np.load(y_path))
with open(vacob_path, 'rb') as f:
    vocab = pickle.load(f)

# X, y, vocab = load_csv(parent_folder, max_seq_len=1000)
print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

vocab_size = len(vocab) + 1
embedding_matrix = get_embedding_matrix(vocab, embedding_dim)
print(f'Embedding Matrix shape: {embedding_matrix.shape}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(vocab_size, embedding_dim, embedding_matrix, nhead, num_encoder_layers, hidden_dim, output_dim).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print(f"Trainable Parameters: {count_parameters(model)}")

for epoch in range(n_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'Ep {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.3f}')
