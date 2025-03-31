import torch
import torch.nn as nn

class UnidirectionalLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, fc1_neurons, output_dim, dropout=0.5):
        super(UnidirectionalLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, fc1_neurons)  # first FC layer
        self.fc2 = nn.Linear(fc1_neurons, output_dim)  # Output layer
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = hidden[-1, :, :]
        hidden = self.relu(self.fc1(hidden))
        output = self.fc2(hidden)
        return output