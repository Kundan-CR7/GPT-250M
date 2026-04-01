import torch
import torch.nn as nn
from torch.nn import functional as F
import math
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
    
class CasualSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        #A single linear layer to calculate Q,K,V simultaneously for all heads
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)

        #Output projection layer (combing multiple attn_head outputs)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)

        #Regularization 
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        #Causal Mask (using register_buffer so PyTorch knows this is part of the model state(not to be trained))
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1,config.block_size, config.block_size))
        
    def forward(self,x):
        B,T,C = x.size()

        #Q,K,V generation
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim=2)   #each shape will be (B,T,C=n_embd)

        #Multi-Head Split
        k = k.view(B,T, self.n_head, C//self.n_head).transpose(1,2)
        q = q.view(B,T, self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B,T, self.n_head, C//self.n_head).transpose(1,2)

        #attention score
        att = (q @ k.transpose(-2,-1) * (1.0 / math.sqrt(k.size(-1))))    #(B,n_head,T,T)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))

        #Softmax
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v     #(B, n_head, T, hs)

        #Putting heads back together (B,T,C)
        y = y.transpose(1,2).contiguous().view(B,T,C)

        #Final output projection
        y = self.resid_dropout(self.c_proj(y))










