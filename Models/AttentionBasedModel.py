import torch
import torch.nn as nn

class AttentionBasedModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim1, hidden_dim2, output_dim, dropout=0.5):
        super(AttentionBasedModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.pos_encoding = self._create_positional_encoding(300, embedding_dim)
        
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_layers)])
        
        self.pool_attention = nn.Linear(embedding_dim, 1)
        
        self.fc1 = nn.Linear(embedding_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def _create_positional_encoding(self, max_length, embedding_dim):
        """Generate positional encodings."""
        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_length, embedding_dim)
    
    def forward(self, text, device=None):  
        if device is None:
            device = torch.device('cpu')  
        embedded = self.embedding(text)
        pe = self.pos_encoding[:, :text.size(1), :].to(device)  
        embedded = embedded + pe
        embedded = self.dropout(embedded)
        
        embedded = embedded.permute(1, 0, 2)
        for attn, norm in zip(self.attn_layers, self.norms):
            attn_output, _ = attn(embedded, embedded, embedded)
            embedded = norm(attn_output + embedded)
        attn_output = embedded.permute(1, 0, 2)
        
        weights = torch.softmax(self.pool_attention(attn_output), dim=1)
        pooled = torch.sum(attn_output * weights, dim=1)
        
        hidden1 = self.relu(self.fc1(pooled))
        hidden2 = self.relu(self.fc2(hidden1))
        output = self.fc3(hidden2)
        return output