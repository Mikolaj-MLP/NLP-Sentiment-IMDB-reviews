import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_sizes, num_filters, output_dim, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)  # +1 for index fix
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))  # (batch_size, max_length, embedding_dim)
        embedded = embedded.permute(0, 2, 1)          # (batch_size, embedding_dim, max_length)
        conv_outputs = [self.relu(conv(embedded)) for conv in self.convs]  # List of (batch_size, num_filters, L)
        pooled = [torch.max(conv_out, dim=2)[0] for conv_out in conv_outputs]  # Max pool: (batch_size, num_filters)
        cat = self.dropout(torch.cat(pooled, dim=1))  # (batch_size, num_filters * len(filter_sizes))
        output = self.fc(cat)                         # (batch_size, output_dim)
        return output