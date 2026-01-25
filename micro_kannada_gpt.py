import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

batch_size = 32        # Number of sequences processed in parallel
block_size = 128       # Maximum context length (T)
n_embd = 64           # Embedding dimension (C)
n_head = 4            # Number of attention heads
n_layer = 4           # Number of transformer blocks
dropout = 0.1         # Dropout probability
learning_rate = 3e-4  # AdamW learning rate
max_iters = 2000      # Training iterations
eval_interval = 100   # Print loss every N iterations
eval_iters = 20       # Average loss over N batches for evaluation
device = 'cuda' if torch.cude.is_available() else 'cpu'

torch.manual_seed(1337)
