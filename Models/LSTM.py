import torch
import torch.nn as nn

class UnidirectionalLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.5):
        super(UnidirectionalLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))  # (batch_size, max_length, embedding_dim)
        output, (hidden, cell) = self.lstm(embedded)   # hidden: (1, batch_size, hidden_dim)
        hidden = hidden.squeeze(0)                     # (batch_size, hidden_dim)
        return self.fc(hidden)                         # (batch_size, output_dim)