import pandas as pd
import numpy as np
import os
import torch
import pickle
import csv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils import CLSTMModel, get_embedding_matrix, train, evaluate

parent_folder = './dataset/'
X_path = os.path.join(parent_folder, 'X.npy')
y_path = os.path.join(parent_folder, 'y.npy')
vocab_path = os.path.join(parent_folder, 'vocab.pkl')

embedding_dim = 300
hidden_dim = 256
output_dim = 1
n_epochs = 10
lr = 0.001

# Load data
X, y = torch.LongTensor(np.load(X_path)), torch.LongTensor(np.load(y_path))
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

vocab_size = len(vocab) + 1
embedding_matrix = get_embedding_matrix(vocab, embedding_dim)

# Experiment with different training set sizes
training_sizes = [0.2, 0.4, 0.6, 0.8]
results = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize CSV file for epoch-wise results
epoch_results_path = os.path.join(parent_folder, 'epoch_wise_results.csv')
with open(epoch_results_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Train Size', 'Epoch', 'Train Loss', 'Test Loss', 'Test Accuracy'])

for train_size in training_sizes:
    # Split the data with a fixed test size of 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=0.2, random_state=42)
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    # Initialize model, criterion, and optimizer
    model = CLSTMModel(vocab_size, embedding_dim, embedding_matrix, hidden_dim, output_dim).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train the model
    for epoch in range(n_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f'Train Size: {train_size}, Epoch: {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.3f}')
        
        # Save epoch-wise results
        with open(epoch_results_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([train_size, epoch + 1, train_loss, test_loss, test_acc])

    # Evaluate final performance
    final_train_loss = train_loss
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    results.append([train_size, final_train_loss, final_test_loss, final_test_acc])

# Save final results to CSV
results_path = os.path.join(parent_folder, 'training_size_experiment.csv')
with open(results_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Train Size', 'Final Train Loss', 'Final Test Loss', 'Final Test Accuracy'])
    writer.writerows(results)

print("Experiment completed. Results saved to training_size_experiment.csv and epoch_wise_results.csv.")

