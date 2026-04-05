import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from config import GPTConfig

class GPTEmbedding(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, idx):
        B,T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embedding(idx)  #(B,T,C)
        pos_emb = self.position_embedding(pos)  #(T,C)
        x = tok_emb + pos_emb   #(B,T,C) broadcasts works here
        return self.drop(x)
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1,config.block_size, config.block_size))
        
    def forward(self, x, attention_mask=None): # <-- Added attention_mask
        B,T,C = x.size()

        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd,dim=2)   

        k = k.view(B,T, self.n_head, C//self.n_head).transpose(1,2)
        q = q.view(B,T, self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B,T, self.n_head, C//self.n_head).transpose(1,2)

        att = (q @ k.transpose(-2,-1) * (1.0 / math.sqrt(k.size(-1))))    
        
        # 1. Apply the standard causal mask (future tokens)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))

        # 2. Apply the padding mask (if provided by DPO dataloader)
        if attention_mask is not None:
            # Reshape from (B, T) to (B, 1, 1, T) so it broadcasts across heads and sequence length
            attention_mask = attention_mask.view(B, 1, 1, T)
            att = att.masked_fill(attention_mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v     
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.resid_dropout(self.c_proj(y))

        return y

class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)    
        x = self.gelu(x)    
        x = self.c_proj(x)  
        x = self.dropout(x)     
        return x
    
class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config=config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config=config)

    def forward(self, x, attention_mask=None): # <-- Added attention_mask
        # Pass the mask into the attention mechanism
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.config = config
        self.embeddings = GPTEmbedding(config)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.embeddings.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if(isinstance(module, nn.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if(module.bias is not None):
                torch.nn.init.zeros_(module.bias)
        elif(isinstance(module, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, attention_mask=None): # <-- Added attention_mask
        x = self.embeddings(idx)   

        # Thread the mask through all 16 Transformer Blocks
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)   

        return logits