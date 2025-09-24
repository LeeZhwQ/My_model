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

d_model = 512
context_length = 128
num_heads = 8
head_size = d_model // num_heads
device = 'cuda' if torch.cuda.is_available() else "cpu"
dropout = 0.1
num_blocks = 12

checkpoint = torch.load("model_v2.ckpt",map_location= torch.device('cpu'))
model = Model()
model.load_state_dict(model.state_dict(checkpoint))
model.eval()
model.to(device)

tokenizer = tiktoken.get_encoding("cl100k_base")

start = "大堂很大"
input_ids = tokenizer.encode(start)
x = (torch.tensor(input_ids,dtype=torch.long,device=device)[None,...])

with torch.no_grad():
    y = model.generate(x,100,1.0)
    print("------------")
    print(tokenizer.decode(y[0].tolist()))
    print("------------")