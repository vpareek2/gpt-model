import torch
import torch.nn as nn
from torch.nn import functional as F

from model_utils import Block


class GPTLanguageModel(nn.Module):
    """
    An implementation of a GPT (Generative Pretrained Transformer) Language Model.
    """
    
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, device):
        """
        Initializes the GPT language model.

        Parameters:
        - vocab_size (int): Size of the vocabulary.
        - n_embd (int): Embedding size.
        - block_size (int): Length of input sequences.
        - n_head (int): Number of heads in multiheadattention models.
        - n_layer (int): Number of transformer blocks.
        - device (str): Device to which tensors will be moved.
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.device = device
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initializes weights for linear and embedding layers.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        """
        Forward pass through the model.

        Parameters:
        - index (torch.Tensor): Input tensor with token ids. Shape: (batch_size, seq_len).
        - targets (torch.Tensor, optional): Target token ids. Shape: (batch_size, seq_len).

        Returns:
        - logits (torch.Tensor): Logits of token predictions. Shape: (batch_size, seq_len, vocab_size).
        - loss (torch.Tensor, optional): Cross-entropy loss. Returned only if `targets` is provided.
        """
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        
        x = tok_emb + pos_emb.unsqueeze(0).expand(B, -1, -1)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    def generate(self, index, max_new_tokens):
        """
        Generates new tokens following the context provided in `index`.

        Parameters:
        - index (torch.Tensor): Context token ids. Shape: (batch_size, seq_len).
        - max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
        - index (torch.Tensor): Generated token ids. Shape: (batch_size, seq_len + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            logits, _ = self.forward(index)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
            
        return index

