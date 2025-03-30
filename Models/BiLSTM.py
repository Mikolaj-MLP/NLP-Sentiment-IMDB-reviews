import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.5):
        super(BidirectionalLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size +1, embedding_dim, padding_idx=0) # +1 is the index fix
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for forward + backward
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))  # (batch_size, max_length, embedding_dim)
        output, (hidden, cell) = self.lstm(embedded)   # hidden: (2, batch_size, hidden_dim)
        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # (batch_size, hidden_dim * 2)
        return self.fc(hidden)                           # (batch_size, output_dim)