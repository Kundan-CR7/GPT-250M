from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024    
    vocab_size: int = 50257      
    n_layer: int = 16            
    n_head: int = 16             
    n_embd: int = 1024          
    dropout: float = 0.1        