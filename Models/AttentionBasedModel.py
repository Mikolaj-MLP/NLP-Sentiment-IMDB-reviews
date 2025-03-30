import torch
import torch.nn as nn

class AttentionBasedModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, output_dim, dropout=0.5):
        super(AttentionBasedModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)  # +1 for index fix
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))  # (batch_size, max_length, embedding_dim)
        # Attention expects (seq_len, batch_size, embed_dim), therefore permute
        embedded = embedded.permute(1, 0, 2)          # (max_length, batch_size, embedding_dim)
        attn_output, _ = self.attention(embedded, embedded, embedded)  # Self-attention
        attn_output = attn_output.permute(1, 0, 2)    # (batch_size, max_length, embedding_dim)
        pooled = torch.mean(attn_output, dim=1)       # Average over sequence length
        hidden = self.relu(self.fc1(pooled))          # (batch_size, hidden_dim)
        output = self.fc2(hidden)                     # (batch_size, output_dim)
        return output