import os
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import CLSTMModel, get_embedding_matrix, evaluate

# Paths for the new dataset (leave blank for user to fill in)
new_dataset_folder = './dataset_1/'  # Provide path to the new dataset folder
X_new_path = os.path.join(new_dataset_folder, 'X.npy')
y_new_path = os.path.join(new_dataset_folder, 'y.npy')
vocab_new_path = os.path.join(new_dataset_folder, 'vocab.pkl')

# Paths for the original model
parent_folder = './dataset/'
model_path = os.path.join(parent_folder, 'model.pth')
vocab_path = os.path.join(parent_folder, 'vocab.pkl')

embedding_dim = 300
hidden_dim = 256
output_dim = 1

# Load original vocabulary
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

vocab_size = len(vocab) + 1
embedding_matrix = get_embedding_matrix(vocab, embedding_dim)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLSTMModel(vocab_size, embedding_dim, embedding_matrix, hidden_dim, output_dim).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Load new dataset
X_new = torch.LongTensor(np.load(X_new_path))
y_new = torch.LongTensor(np.load(y_new_path))

# Replace out-of-vocabulary indices with an 'unknown' token index
UNK_IDX = vocab.get('<UNK>', len(vocab))  # Use len(vocab) as the index for unknown tokens if not predefined
X_new = torch.where(X_new < vocab_size, X_new, torch.tensor(UNK_IDX))

# Create DataLoader for the new dataset
test_data = TensorDataset(X_new, y_new)
test_loader = DataLoader(test_data, batch_size=32)

# Evaluate the model on the new dataset
criterion = torch.nn.BCELoss()
new_test_loss, new_test_acc = evaluate(model, test_loader, criterion, device)

print(f'Test Loss on New Dataset: {new_test_loss:.4f}')
print(f'Test Accuracy on New Dataset: {new_test_acc:.3f}')

