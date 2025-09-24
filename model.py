import torch
import torch.nn as nn
from torch.nn import functional as F
import math

#Hyperparameters
d_model = 512
context_length = 32
num_heads = 8
head_size = d_model // num_heads
device = 'cuda' if torch.cuda.is_available() else "cpu"
dropout = 0.1
num_blocks = 12


class feed_forward_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self,x):
        return self.ffn(x)
    
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(d_model,head_size,bias=False)
        self.Wk = nn.Linear(d_model,head_size,bias=False)
        self.Wv = nn.Linear(d_model,head_size,bias=False)
        self.register_buffer('mask',torch.tril(torch.ones(context_length,context_length)))
        self.Dropout = nn.Dropout(dropout)
        
    def forward(self,x): 
        B , T , D = x.shape
        q = self.Wq(x) # batchsize * context_length * head_size
        k = self.Wk(x)
        v = self.Wv(x)
        
        output = (q @ k.transpose(-2,-1)) / math.sqrt(head_size)
        output =  output.masked_fill(self.mask[:T,:T] == 0 , float('-inf'))
        output = F.softmax(output,dim=-1)
        output = self.Dropout(output)
        
        output = output @ v
        
        return output
    
class Multi_head_Attention(nn.Module):
    def __init__(self):# batchsize * context_length * head_size
        super().__init__()
        self.heads = nn.ModuleList([Attention() for _ in range(num_heads)])
        self.Wo = nn.Linear(d_model, d_model) #Output的Wo
        self.Dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        output = torch.cat([head(x) for head in self.heads],dim= -1)
        output = self.Wo(output)
        output = self.Dropout(output)
        return output

class Transformer_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = Multi_head_Attention()
        self.ffn = feed_forward_network()
        
    def forward(self,x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        
        return x
    
class Model(nn.Module):
    def __init__(self,max_token_value = 100256):
        super().__init__()
        self.vocab_linear = nn.Linear(d_model,max_token_value + 1)
        self.te_lookup_tabel = nn.Embedding(max_token_value + 1,d_model) #embedding 层
        self.transformerBlock =  nn.Sequential(
            *([Transformer_Block() for _ in range(num_blocks)] + [nn.LayerNorm(d_model)]) #将最后的layernorm加上了
        )
         
    def forward(self, x_batch ,y_batch = None): # batchsize * context_length 
        B,T = x_batch.shape
        pe_lookup_tabel = torch.zeros(context_length,d_model,device=device) # context * d_model
        position = torch.arange(0,context_length,dtype= torch.float).unsqueeze(1)
        div_term = torch.exp(- torch.log(10000.0) * torch.arange(0,d_model,2).float / d_model)
        pe_lookup_tabel[: , 0 :: 2] = torch.sin(position * div_term)
        pe_lookup_tabel[:,1::2] = torch.cos(position * div_term)
        
        output = self.te_lookup_tabel(x_batch) + pe_lookup_tabel[:T,:]
        
        output = self.transformerBlock(output)
        
        logits = self.vocab_linear(output)
        
        if y_batch is not None:
            B,T,D = logits.shape
            logits_reshaped = logits.view(B*T,D)
            y_reshaped = y_batch.view(B*T)
            loss = F.cross_entropy(input=logits_reshaped,target=y_reshaped)
        else:
            loss = None
        
        return logits,loss
    
    def generate(self,x_batch,max_new_tokens=100,temerature=1.0):
        for _ in range(max_new_tokens):
            #x_batch: batchsize * context_length
            x_crop = x_batch[:,-context_length:]
            logits,loss = self.forward(x_crop) #logits [batch,T,vocab_size]
            logits = logits[:,-1,:] / temerature
            probability = F.softmax(logits,dim=-1)
            predicted = torch.multinomial(probability,num_samples=1) #是一个索引值
            x_batch = torch.cat((x_batch,predicted),dim=1)
        
        return x_batch
        
        
        
        
        
        
        
        