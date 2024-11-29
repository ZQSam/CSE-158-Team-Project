import pandas as pd
import numpy as np
import re
import os
import math
import torch
# import torchtext; torchtext.disable_torchtext_deprecation_warning()
import torch.nn as nn
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe
from sklearn.utils import shuffle
from tqdm import tqdm


glove = GloVe(name='6B', dim=300)
tokenizer = get_tokenizer('basic_english')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

def build_vocab(texts):
    vocab = {}
    for text in texts:
        tokens = tokenizer(text)
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab) + 1
    return vocab

def text_to_sequence(text, vocab, max_len=300):
    tokens = tokenizer(text)
    seq = [vocab[token] for token in tokens if token in vocab]
    return seq[:max_len] + [0] * (max_len - len(seq))

def get_embedding_matrix(vocab, embedding_dim):
    vocab_size = len(vocab) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in vocab.items():
        embedding_matrix[idx] = glove[word]
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    return embedding_matrix

def load_dataset(parent_folder:str, max_seq_len=300):
    real_data = os.path.join(parent_folder, './True.csv')
    fake_data = os.path.join(parent_folder, './Fake.csv')
    real_df = pd.read_csv(real_data)
    fake_df = pd.read_csv(fake_data)
    real_df['label'] = 0
    fake_df['label'] = 1

    df = pd.concat([real_df, fake_df])
    df = shuffle(df).reset_index(drop=True)
    df['clean_text'] = df['text'].apply(clean_text)
    vocab = build_vocab(df['clean_text'])
    df['sequence'] = df['clean_text'].apply(lambda x: text_to_sequence(x, vocab, max_len=max_seq_len))
    
    X = np.array(df['sequence'].tolist())
    y = np.array(df['label'].values)
    return X, y, vocab

def load_csv(parent_folder:str, max_seq_len=300):
    real_data = os.path.join(parent_folder, './train.csv')
    df = pd.read_csv(real_data)
    df = df.dropna(subset=['text'])

    df['clean_text'] = df['text'].apply(clean_text)
    vocab = build_vocab(df['clean_text'])
    df['sequence'] = df['clean_text'].apply(lambda x: text_to_sequence(x, vocab, max_len=max_seq_len))

    X = np.array(df['sequence'].tolist())
    y = np.array(df['label'].values)
    return X, y, vocab

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, hidden_dim, output_dim):
        super(CLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(embedding_matrix, requires_grad=False)

        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5)
        self.pool = nn.MaxPool1d(kernel_size=4)

        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = x.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len] for Conv1d
        x = self.conv1(x)
        x = self.pool(torch.relu(x))
        x = self.conv2(x)
        x = self.pool(torch.relu(x))

        x = x.permute(0, 2, 1)  # [batch_size, seq_len, conv_output_dim] for LSTM
        x, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])  # Take the hidden state of the last LSTM layer

        output = torch.sigmoid(self.fc(x))
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, 
                 nhead, num_encoder_layers, hidden_dim, output_dim, max_len=1000):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(embedding_matrix, requires_grad=False)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, 
                                                    dim_feedforward=hidden_dim, dropout=0.5, 
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        x = self.embedding(x).to('cuda')  # [batch_size, seq_len, embedding_dim]
        x = self.pos_encoder(x)  # Add positional encoding
        x = self.transformer_encoder(x)  # Pass through transformer encoder
        x = x.mean(dim=1)  # Average over sequence length (pooling) now along dim=1 because batch is first
        x = self.dropout(x)
        output = torch.sigmoid(self.fc(x))
        return output


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy