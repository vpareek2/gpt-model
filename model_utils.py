import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ 
    Head of a multi-head self-attention mechanism.
    
    This module computes the self-attention for a single head, 
    providing the ability to focus on different parts of the input sequence when producing the output.
    """

    def __init__(self, n_embd, head_size, block_size, dropout):
        """
        Initialize the head.
        
        Parameters:
        - n_embd (int): Dimensionality of the input embeddings.
        - head_size (int): Dimensionality of the query, key, and value projections.
        - block_size (int): Maximum sequence length supported by positional embeddings.
        - dropout (float): Dropout rate for the attention weights.
        """
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the self-attention head.
        
        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embd).
        
        Returns:
        - out (torch.Tensor): Output tensor of shape (batch_size, seq_len, head_size).
        """
        B, T, C = x.shape  # batch size, time steps, and channel dimensions
        
        # Compute query, key, and value projections
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        
        # Compute scaled dot-product attention scores
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # scaling by the square root of the key size
        
        # Apply causal (lower triangular) mask to the attention scores
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        
        # Compute softmax along the last dimension to obtain attention weights
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        
        # Compute the weighted sum of value projections to obtain the output
        v = self.value(x) # (B,T,head_size)
        out = wei @ v     # (B, T, head_size)
        
        return out

class MultiHeadAttention(nn.Module):
    """ 
    MultiHeadAttention employs multiple attention heads working in parallel 
    to allow model to focus on different parts of the input sequence when 
    producing the output, capturing various aspects of the input information.
    """

    def __init__(self, num_heads, head_size, n_embd, dropout):
        """
        Initialize the MultiHeadAttention module.
        
        Parameters:
        - num_heads (int): Number of attention heads.
        - head_size (int): Dimensionality of the query, key, and value projections per head.
        - n_embd (int): Dimensionality of the input embeddings.
        - dropout (float): Dropout rate for the attention weights and output projection.
        """
        super().__init__()
        # Initialize all attention heads
        self.heads = nn.ModuleList([Head(n_embd, head_size, n_embd, dropout) for _ in range(num_heads)])
        # Output projection to return to the input dimensionality
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the multi-head self-attention mechanism.
        
        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embd).
        
        Returns:
        - out (torch.Tensor): Output tensor of shape (batch_size, seq_len, n_embd).
        """
        # Concatenate the output of all attention heads along the last dimension
        out = torch.cat([head(x) for head in self.heads], dim=-1) # (B, T, head_size * num_heads)
        
        # Apply the output projection and dropout
        out = self.dropout(self.proj(out)) # (B, T, n_embd)
        
        return out

class FeedForward(nn.Module):
    """ 
    A feedforward neural network consisting of two linear layers and 
    a non-linearity, often used in transformer architectures to 
    transform the representation obtained after the attention mechanism.
    """
    
    def __init__(self, n_embd, dropout):
        """
        Initialize the FeedForward module.
        
        Parameters:
        - n_embd (int): Size of the input embedding vector.
        - dropout (float): Dropout rate for the dropout layer in the network.
        """
        super().__init__()
        
        # Define a simple feedforward network
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Linear layer to expand the input embedding
            nn.ReLU(),  # Non-linearity
            nn.Linear(4 * n_embd, n_embd),  # Linear layer to project back to input embedding size
            nn.Dropout(dropout)  # Dropout layer to prevent overfitting
        )

    def forward(self, x):
        """
        Forward pass through the feedforward network.
        
        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embd).
        
        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embd).
        """
        return self.net(x)


class Block(nn.Module):
    """ 
    A transformer block combining self-attention and feed-forward layers,
    encapsulating the 'communication' followed by 'computation' mechanism
    prevalent in transformer models.
    """
    
    def __init__(self, n_embd, n_head, dropout=0.1):
        """
        Initialize the transformer block.
        
        Parameters:
        - n_embd (int): Size of the embedding.
        - n_head (int): Number of attention heads in the multi-head attention mechanism.
        - dropout (float, optional): Dropout rate for layers; default is 0.1.
        """
        super().__init__()
        
        # Ensure embedding size is divisible by number of heads
        assert n_embd % n_head == 0, "Embedding size must be divisible by number of heads"
        
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """
        Forward pass through the transformer block.
        
        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embd).
        
        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, seq_len, n_embd).
        """
        y = self.sa(x)  # Apply self-attention
        x = self.ln1(x + y)  # Add & norm
        y = self.ffwd(x)  # Feed-forward
        x = self.ln2(x + y)  # Add & norm
        return x
