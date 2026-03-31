import torch
import torch.nn as nn
from config import GPTConfig

class GPTEmbedding(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        self.position_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        self.drop = nn.Dropout(config.dropout)

    def forward(self, idx):
        B,T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embedding(idx)  #(B,T,C)
        pos_emb = self.position_embedding(pos)  #(T,C)

        x = tok_emb + pos_emb   #(B,T,C) broadcasts works here
        return self.drop(x)