import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_sizes=[2, 3, 4, 5], num_filters=128, 
                 fc1_neurons=256, fc2_neurons=128, output_dim=2, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, fc1_neurons)  # first FC layer
        self.fc2 = nn.Linear(fc1_neurons, fc2_neurons)                      # second FC layer
        self.fc3 = nn.Linear(fc2_neurons, output_dim)                       # Output layer
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        embedded = embedded.permute(0, 2, 1)
        conv_outputs = [self.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max(conv_out, dim=2)[0] for conv_out in conv_outputs]
        cat = self.dropout(torch.cat(pooled, dim=1))
        hidden = self.relu(self.fc1(cat))
        hidden = self.relu(self.fc2(hidden))
        output = self.fc3(hidden)
        return output