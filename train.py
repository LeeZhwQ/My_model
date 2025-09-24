import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import os
import sys
import pickle
from model import Model
import tiktoken
import pandas as pd

#Hyperparameters
d_model = 512
context_length = 128
num_heads = 8
head_size = d_model // num_heads
device = 'cuda' if torch.cuda.is_available() else "cpu"
dropout = 0.1
num_blocks = 12
batch_size = 12
max_steps = 100000
learning_rate = 1e-3
eval_interval = 20
eval_iters = 5
TORCH_SEED = 1072
torch.manual_seed(TORCH_SEED)


df = pd.read_csv("/home/zhenghao/My_model/ChnSentiCorp_htl_all.csv")
all_reviews = df["review"].tolist()  
text = ";".join(str(review) for review in all_reviews)

tokenizer = tiktoken.get_encoding("cl100k_base")
tokenized_text = tokenizer.encode(text)
tokenizer_text = torch.tensor(tokenized_text,dtype = torch.long ,device=device)

train_size = int(len(text) * 0.9)
train_data = torch.tensor(tokenized_text[:train_size], dtype=torch.long, device=device)
vali_data = torch.tensor(tokenized_text[train_size:], dtype=torch.long, device=device)

def get_batch(split):
    data = train_data if split == "train" else vali_data
    idxs = torch.randint(low=0,high=len(data)-context_length,size=(batch_size,))
    x = torch.stack([data[idx:idx+context_length] for idx in idxs])
    y = torch.stack([data[idx + 1 : idx + 1 + context_length] for idx in idxs])
    
    return x,y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train','validation']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            _,loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = Model().to(device)
ckpt_path = os.path.join(os.path.dirname(__file__), 'model.ckpt')
model.load_state_dict(torch.load(ckpt_path, map_location=device))
optimizer = torch.optim.Adamax(model.parameters(),lr=learning_rate)
for step in range(max_steps):
    
    if step % eval_interval == 0 or step == max_steps - 1:
        out = estimate_loss(model=model)
        print('Step',step,' Train_loss: ',round(out['train'].item(),3),' Validation_loss: ',round(out['validation'].item(),3))
    
    x,y = get_batch('train')
    logits,loss = model(x,y)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    
    
torch.save(model.state_dict(),'model_v2.ckpt')
