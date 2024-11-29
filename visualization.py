import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import torch
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils import *

parent_folder = './dataset/'
metrics_path = os.path.join(parent_folder, 'metrics.csv')
model_path = os.path.join(parent_folder, 'model.pth')
X_test_path = os.path.join(parent_folder, 'X.npy')
y_test_path = os.path.join(parent_folder, 'y.npy')
vocab_path = os.path.join(parent_folder, 'vocab.pkl')

# Load the metrics CSV
metrics_df = pd.read_csv(metrics_path)

# Set global style for plots
sns.set(style='whitegrid')

# Plot Training Loss vs Epochs
plt.figure(figsize=(10, 6))
sns.lineplot(x='Epoch', y='Train Loss', data=metrics_df, marker='o', label='Training Loss', linewidth=2.5)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Training Loss over Epochs', fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(parent_folder, 'visualization/training_loss.png'))
plt.show()

# Plot Test Loss vs Epochs
plt.figure(figsize=(10, 6))
sns.lineplot(x='Epoch', y='Test Loss', data=metrics_df, marker='o', color='orange', label='Test Loss', linewidth=2.5)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.title('Test Loss over Epochs', fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(parent_folder, 'visualization/test_loss.png'))
plt.show()

# Plot Test Accuracy vs Epochs
plt.figure(figsize=(10, 6))
sns.lineplot(x='Epoch', y='Test Accuracy', data=metrics_df, marker='o', color='green', label='Test Accuracy', linewidth=2.5)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('Test Accuracy over Epochs', fontsize=16)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(parent_folder, 'visualization/test_accuracy.png'))
plt.show()

# Load vocab to get correct vocab size and embedding matrix
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

vocab_size = len(vocab) + 1
embedding_matrix = get_embedding_matrix(vocab, 300)

# Load model and test data
model = CLSTMModel(vocab_size=vocab_size, embedding_dim=300, embedding_matrix=embedding_matrix, hidden_dim=256, output_dim=1)
model.load_state_dict(torch.load(model_path))
model.eval()

X_test = torch.LongTensor(np.load(X_test_path))
y_test = torch.LongTensor(np.load(y_test_path))

# Get predictions
with torch.no_grad():
    y_pred_probs = model(X_test).squeeze()
    y_pred = (y_pred_probs >= 0.5).long()

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
cmd.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(parent_folder, 'visualization/confusion_matrix.png'))
plt.show()

