# Micro-KannadaGPT
Micro-KannadaGPT from scratch using PyTorch. The goal is to demonstrate a deep understanding of Transformer internals, implementing the architecture manually without using nn.Transformer or HuggingFace.

🏗️ Architecture Overview
Our Model: Decoder-Only Transformer (GPT-style)
```
Input Text: "ಜ್ಞಾನವೇ"
      ↓
┌─────────────────────────────────────────────────────┐
│  CHARACTER TOKENIZER                                │
│  "ಜ್ಞಾನವೇ" → [23, 45, 18, 67, 34, 56, 78]        │
└─────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────┐
│  EMBEDDING LAYERS                                   │
│  • Token Embedding: [23] → [0.12, -0.34, ..., 0.56]│
│  • Position Embedding: [0] → [0.23, 0.11, ..., -0.2]│
│  • Combined: Element-wise addition                 │
└─────────────────────────────────────────────────────┘
      ↓ (B, T, 64)
┌─────────────────────────────────────────────────────┐
│  DECODER BLOCK 1                                    │
│  ┌───────────────────────────────────────────────┐  │
│  │ LayerNorm(x)                                  │  │
│  │ Causal Self-Attention (4 heads)              │  │
│  │ x = x + attention_output  ← Residual         │  │
│  └───────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────┐  │
│  │ LayerNorm(x)                                  │  │
│  │ Feed-Forward Network (64→256→64)             │  │
│  │ x = x + ffn_output  ← Residual               │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
      ↓ (B, T, 64)
      ... [Blocks 2, 3, 4] ...
      ↓ (B, T, 64)
┌─────────────────────────────────────────────────────┐
│  FINAL LAYER NORM                                   │
└─────────────────────────────────────────────────────┘
      ↓ (B, T, 64)
┌─────────────────────────────────────────────────────┐
│  LINEAR (LM HEAD): 64 → vocab_size (45)            │
└─────────────────────────────────────────────────────┘
      ↓ (B, T, 45)
┌─────────────────────────────────────────────────────┐
│  SOFTMAX → Probability Distribution                 │
│  [0.02, 0.15, 0.01, ..., 0.08] (sums to 1.0)      │
└─────────────────────────────────────────────────────┘
      ↓
    SAMPLE → Next Token
```
Model Specifications
```
Parameter Value Purpose
vocab_size 45 Number of unique characters in Kannada t
extn_embd 64 Embedding dimension (vector size per token)
n_head 4 Number of parallel attention heads
n_layer 4 Number of decoder blocks stacked
block_size 128 Maximum sequence length (context window)
dropout 0.1 Regularization probability
total_params ~209 KTotal trainable parameters
```

🧩 Core Components Explained
1. Causal Self-Attention
Purpose: Allow each token to gather information from previous tokens while preventing future information leakage.
Mathematical Formula:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) @ V

Where:
  Q = Query matrix  = X @ W_q   (What am I looking for?)
  K = Key matrix    = X @ W_k   (What do I contain?)
  V = Value matrix  = X @ W_v   (What do I communicate?)
  d_k = head_size (scaling factor)
```
Step-by-Step Process:
```
python# Input: x of shape (B, T, C) where B=batch, T=time, C=channels

# Step 1: Project to Q, K, V
Q = x @ W_q  # (B, T, head_size)
K = x @ W_k  # (B, T, head_size)
V = x @ W_v  # (B, T, head_size)

# Step 2: Compute attention scores
scores = Q @ K.transpose(-2, -1)  # (B, T, T)
# Each row i, column j = "How much should token i attend to token j?"

# Step 3: Scale
scores = scores / sqrt(head_size)
# Prevents softmax saturation for large d_k

# Step 4: Apply causal mask
mask = tril(ones(T, T))  # Lower triangular matrix
scores = scores.masked_fill(mask == 0, -inf)
# Sets future positions to -infinity

# Example for T=4:
# Before mask:        After mask:
# [[0.8, 0.6, 0.4, 0.2]    [[0.8, -inf, -inf, -inf]
#  [0.7, 0.9, 0.5, 0.3]     [0.7,  0.9, -inf, -inf]
#  [0.5, 0.6, 0.8, 0.4]     [0.5,  0.6,  0.8, -inf]
#  [0.4, 0.5, 0.6, 0.9]]    [0.4,  0.5,  0.6,  0.9]]

# Step 5: Softmax (makes each row sum to 1.0)
weights = softmax(scores, dim=-1)  # (B, T, T)
# After softmax:
# [[1.00, 0.00, 0.00, 0.00]   ← Token 0 attends 100% to itself
#  [0.45, 0.55, 0.00, 0.00]   ← Token 1: 45% to tok0, 55% to itself
#  [0.25, 0.30, 0.45, 0.00]   ← Token 2 distributes attention
#  [0.20, 0.25, 0.28, 0.27]]  ← Token 3 attends to all previous

# Step 6: Weighted sum of values
output = weights @ V  # (B, T, head_size)
# Token at position i gets: weighted average of values from positions 0..i
```
