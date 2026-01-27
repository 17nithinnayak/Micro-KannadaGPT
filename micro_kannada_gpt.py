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

kannada_text = """
ಹಸಿರು ಎಲೆಗಳ ನಡುವೆ ಹೂವು ಅರಳುತ್ತದೆ.
ನದಿಯ ಹರಿವು ಎಂದಿಗೂ ನಿಲ್ಲುವುದಿಲ್ಲ.
ಜ್ಞಾನವೇ ಶ್ರೇಷ್ಠ ಸಂಪತ್ತು.
ಸತ್ಯವೇ ದೇವರು, ಧರ್ಮವೇ ಜೀವನ.
ಮರದ ನೆರಳಿನಲ್ಲಿ ಪಕ್ಷಿಗಳು ಹಾಡುತ್ತವೆ.
ಬೆಳಗಿನ ಸೂರ್ಯ ಎಲ್ಲರಿಗೂ ಸಮಾನ.
ಪ್ರೀತಿ ಎಲ್ಲ ಬಾಧೆಗಳನ್ನು ದಾಟುತ್ತದೆ.
ಕಲಿಕೆಯಲ್ಲಿ ವಯಸ್ಸಿಲ್ಲ.
ಶಾಂತಿ ಮನಸ್ಸಿನಲ್ಲಿ ಹುಟ್ಟುತ್ತದೆ.
ನಮ್ರತೆಯೇ ಶ್ರೇಷ್ಠ ಗುಣ.
ಹೂವಿನ ಸುಗಂಧ ದೂರ ಹರಡುತ್ತದೆ.
ಕಷ್ಟವೇ ಯಶಸ್ಸಿಗೆ ದಾರಿ.
ಸಹಾನುಭೂತಿ ಮಾನವೀಯತೆಯ ಮೂಲ.
ಕನಸುಗಳು ನಿಜವಾಗಲು ಶ್ರಮ ಬೇಕು.
ಪ್ರಕೃತಿಯೇ ನಮ್ಮ ಗುರು.
ಮಳೆಯ ಹನಿಗಳು ಭೂಮಿಯನ್ನು ತಣಿಸುತ್ತವೆ.
ನಗು ಅತ್ಯುತ್ತಮ ಔಷಧ.
ಸಮಯವೇ ಅಮೂಲ್ಯ ಧನ.
ಸ್ನೇಹವೇ ಜೀವನದ ಆಧಾರ.
ತಾಳ್ಮೆಯಿಂದ ಎಲ್ಲವೂ ಸಾಧ್ಯ.
ಪುಸ್ತಕಗಳು ಜ್ಞಾನದ ಭಂಡಾರ.
ಆಕಾಶವು ಮಿತಿಯಿಲ್ಲದ ವಿಸ್ತಾರ.
ಸಂಗೀತ ಮನಸ್ಸಿಗೆ ಆಹಾರ.
ಚಂದ್ರನ ಬೆಳಕು ರಾತ್ರಿಯನ್ನು ಬೆಳಗಿಸುತ್ತದೆ.
ಕನ್ನಡವೇ ನಮ್ಮ ಹೆಮ್ಮೆಯ ಭಾಷೆ.
ಎಲೆಗಳ ಸಪ್ಪಳ ಗಾಳಿಯ ಹಾಡು.
ಸಾಗರದ ಅಲೆಗಳು ನಿರಂತರ ಚಲನೆ.
ಬೆಳಗಾವಲು ಹೊಸ ಭರವಸೆಯ ಸಂಕೇತ.
ತಾರೆಗಳು ಆಕಾಶದ ಆಭರಣ.
ಮಾತು ಮನಸ್ಸಿನ ಕನ್ನಡಿ.
"""

class CharTokenizer:
    def __init__(self, text):
       chars = sorted(list(set(text)))
       self.vocab_size = len(chars)
       self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
       self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
       print(f"Tokenizer initialized: {self.vocab_size} unique characters")
       print(f"Sample chars: {chars[:10]}")

    def encode(self, text):
        """Convert string to list of integers"""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        """Convert list of integers back to string"""
        return ''.join([self.idx_to_char[i] for i in indices])

