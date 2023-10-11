# Importing necessary libraries and modules
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle

from args import get_args
from text_utils import load_chars, get_mappings, encode, decode
from data_utils import get_batch
from model import GPTLanguageModel

file_path = "openwebtext/vocab.txt"

batch_size = get_args['batch_size']
block_size = get_args['block_size']
max_iters = get_args['max_iters']
learning_rate = get_args['learning_rate']
eval_iters = get_args['eval_iters']
n_embd = get_args['n_embd']
n_head = get_args['n_head']
n_layer = get_args['n_layer']
dropout = get_args['dropout']

# Determine the computing device.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

chars = load_chars(file_path)
stoi, itos = get_mappings(chars)

encoded = lambda s: encode(stoi, s)
decoded = lambda l: decode(itos, l)

train_file_path = "openwebtext/train_split.txt"  # Adjust path as needed
x, y = get_batch(encoded, train_file_path, block_size, batch_size, device)


@torch.no_grad()  # Decorator to disable gradient computation during inference, saving memory
def estimate_loss():
    out = {}  # Initialize an empty dictionary to hold the losses
    model.eval()  # Set the model to evaluation mode, affecting certain layers like dropout layers

    # Loop over the 'train' and 'val' data splits
    for split in ['train', 'val']:
        # Initialize a tensor to hold the loss values during evaluation iterations
        losses = torch.zeros(eval_iters)

        # Loop over the specified number of evaluation iterations
        for k in range(eval_iters):
            # Get a batch of data
            X, Y = get_batch(split)
            # Get the model's predictions and calculate the loss
            logits, loss = model(X, Y)
            # Store the current iteration's loss
            losses[k] = loss.item()

        # Calculate the mean loss and store it in the 'out' dictionary
        out[split] = losses.mean()

    # Set the model back to training mode
    model.train()

    return out

# Initialize the model and move it to the GPU (if available)
model = GPTLanguageModel()
model = model.to(device)  # Ensure the model is on the same device as the input tensors will be

# Uncomment below if you want to load a pre-trained model
# print('loading model parameters...')
# with open('model-01.pkl', 'rb') as f:
#     model = pickle.load(f).to(device)  # Ensure loaded model is on the correct device
# print('loaded successfully!')

# Initialize the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    print(f"Iteration: {iter}")

    # Evaluation and logging
    if iter % eval_iters == 0:
        losses = estimate_loss()  # Assume this function is defined and returns loss
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

    # Sample a batch of data
    xb, yb = get_batch('train')  # Assume `get_batch` is defined and returns a training batch
    xb, yb = xb.to(device), yb.to(device)  # Ensure data is on the same device as the model
    
    # Forward pass
    logits, loss = model(xb, yb)  # Direct call, assuming `forward` method is defined in the model
    
    # Backward pass and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    print(f"Loss: {loss.item():.3f}")

# Save the trained model
with open('model-01.pkl', 'wb') as f:
    pickle.dump(model.cpu(), f)  # Move model to CPU to avoid issues with pickle and CUDA

print('Model saved.')
