import torch.nn as nn 
import torch.nn.functional as F 
from dataclasses import dataclass
import torch 
import math 
from transformers import GPT2LMHeadModel


@dataclass
class GPTConfig:
    n_embd : int      = 768
    vocab_size : int  = 50257
    n_head : int      = 12
    block_size : int  = 1024 
    n_blocks  : int   = 12 


class MLPLayer(nn.Module):
    def __init__(self, config : GPTConfig):
        super().__init__()
        self.c_fc   = nn.Linear(in_features=config.n_embd, out_features=config.n_embd *4)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(in_features=config.n_embd *4, out_features=config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x 
    
class SelfAttention(nn.Module):
    def __init__(self, config : GPTConfig):
        super().__init__()
        self.config = config
        self.c_attn  = nn.Linear(in_features=config.n_embd, out_features=config.n_embd *3)
        self.c_proj  = nn.Linear(in_features=config.n_embd, out_features=config.n_embd)
        tril_ = torch.tril(torch.ones(config.block_size, config.block_size).view(1,1,config.block_size, config.block_size))
        self.register_buffer('bias', tril_)
    def forward(self, x : torch.tensor):
        qkv    = self.c_attn(x) # (B,T, dmodel) -> (B,T, dmodel *3)
        q,k,v = qkv.chunk(3, dim=-1)  # (B,T, dmodel*3) -> [(B,T dmodel)] *3 
        (B,T,d_model) = q.size()
        h = self.config.n_head
        d = d_model//self.config.n_head
        q = q.view(B,T,h, d ).transpose(1,2)
        k = k.view(B,T,h, d ).transpose(1,2)
        v = v.view(B,T,h, d ).transpose(1,2)
        att = q @ k.transpose(-1,-2) * (1/math.sqrt(d))  # (B,h,T,d) @ (B,h,d,T) -> (B,h,T,T)
        att = att.masked_fill(self.bias[:,:,:T, :T]== 0, float('-inf'))
        out = F.softmax(att, dim=-1) @ v   # (B, h, T,T) @ (B,h, T, d)-> (B,h, T, d)
        out = out.transpose(1,2).contiguous().view(B,T,d_model) #(B,h, T,  d) -> (B, T, d_model)
        out = self.c_proj(out)
        return out 
    


class AttentionWrapper(nn.Module):
    def __init__(self,config : GPTConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config=config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLPLayer(config)

    def forward(self, x : torch.Tensor):
        x = x + self.attn(self.ln_1(x)) 
        x = x + self.mlp(self.ln_2(x))

        return x 


class GPTModel(nn.Module):
    def __init__(self,config : GPTConfig):
        super().__init__()
        self.transformer = nn.ModuleDict(modules={
            "wte" : nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.n_embd),
            "wpe" : nn.Embedding(num_embeddings=config.block_size, embedding_dim=config.n_embd),
            "h"   : nn.ModuleList([ AttentionWrapper(config=config) for _ in range(config.n_blocks)]),
            "ln_f" : nn.LayerNorm(config.n_embd)
         })
        
        self.lm_head = nn.Linear(in_features=config.n_embd, out_features=config.vocab_size,bias=False)

        
    def forward(self, in_tok ):
        (B,T) = in_tok.size()
        pos = torch.arange(0, T, dtype=torch.long, device=in_tok.device)
        tok_emb   = self.transformer.wte(in_tok) 
        pos_emb   = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h : 
            x = block(x)
        x  = self.transformer.ln_f(x)
        logits  = self.lm_head(x)

        return logits 
    

    @classmethod
    def load_pretrained_weights(cls):

        gpt_config = GPTConfig()
        model = GPTModel(config=gpt_config)

        gpt_hf = GPT2LMHeadModel.from_pretrained('gpt2')
        pre_trained_weights  = gpt_hf.state_dict()

        state_dict_torch = model.state_dict()
        py_keys = [key for key in state_dict_torch.keys() if not key.endswith('.attn.bias')]
        hf_keys = [ key for key in pre_trained_weights.keys()  if not key.endswith('.attn.masked_bias')  ]

        transpose_tensors = ['.attn.c_attn.weight','.attn.c_attn.weight', '.mlp.c_fc.weight', '.mlp.c_proj.weight']
        print("copying the data")
        for key in hf_keys:
            list_check = [key.endswith(t_tensor) for t_tensor in transpose_tensors]
            if True in list_check:
                assert pre_trained_weights[key].shape[::-1] == state_dict_torch[key].shape

                with torch.no_grad():
                    state_dict_torch[key].copy_(pre_trained_weights[key].t())
            
            else:
                assert pre_trained_weights[key].shape == state_dict_torch[key].shape
                with torch.no_grad():
                    state_dict_torch[key].copy_(pre_trained_weights[key])


        return model, pre_trained_weights













