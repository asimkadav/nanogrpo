import os
import pickle
import torch
import numpy as np

# Load the dataset
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
train_data = np.memmap(os.path.join(os.path.dirname(__file__), 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(os.path.dirname(__file__), 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(block_size, split='train'):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (64,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x, y