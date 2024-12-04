import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

parent_folder = './dataset/'
epoch_results_path = os.path.join(parent_folder, 'epoch_wise_results.csv')

# Load the epoch-wise results CSV
epoch_results_df = pd.read_csv(epoch_results_path)

# Set global style for plots
sns.set(style='whitegrid', palette='colorblind')

# Define custom color palette
custom_palette = sns.color_palette('tab10')

# Plot Training Loss vs Epochs for Different Training Sizes
plt.figure(figsize=(12, 8))
sns.lineplot(data=epoch_results_df, x='Epoch', y='Train Loss', hue='Train Size', marker='o', linewidth=2.5, palette=custom_palette)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Train Loss', fontsize=14)
plt.title('Training Loss over Epochs for Different Training Sizes', fontsize=16)
plt.legend(title='Train Size', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(parent_folder, 'visualization/train_loss_epoch_vs_size.png'))
plt.show()

# Plot Test Loss vs Epochs for Different Training Sizes
plt.figure(figsize=(12, 8))
sns.lineplot(data=epoch_results_df, x='Epoch', y='Test Loss', hue='Train Size', marker='o', linewidth=2.5, palette=custom_palette)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Test Loss', fontsize=14)
plt.title('Test Loss over Epochs for Different Training Sizes', fontsize=16)
plt.legend(title='Train Size', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(parent_folder, 'visualization/test_loss_epoch_vs_size.png'))
plt.show()

# Plot Test Accuracy vs Epochs for Different Training Sizes
plt.figure(figsize=(12, 8))
sns.lineplot(data=epoch_results_df, x='Epoch', y='Test Accuracy', hue='Train Size', marker='o', linewidth=2.5, palette=custom_palette)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Test Accuracy', fontsize=14)
plt.title('Test Accuracy over Epochs for Different Training Sizes', fontsize=16)
plt.legend(title='Train Size', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(parent_folder, 'visualization/test_accuracy_epoch_vs_size.png'))
plt.show()

