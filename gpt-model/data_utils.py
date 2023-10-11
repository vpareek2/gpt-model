import torch
import random
import mmap

def get_random_chunk(encode_func, file_path, block_size, batch_size):
    """
    Retrieve a random chunk of data from a file.

    Args:
    - encode_func: Function to encode text to integer.
    - file_path: Path to the file.
    - block_size: Size of the data block to read.
    - batch_size: Batch size.

    Returns:
    - data: PyTorch tensor with encoded data block.
    """
    with open(file_path, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
    return torch.tensor(encode_func(decoded_block), dtype=torch.long)

def get_batch(encode_func, file_path, block_size, batch_size, device):
    """
    Retrieve a batch of data.

    Args:
    - encode_func: Function to encode text to integer.
    - file_path: Path to the file.
    - block_size: Size of the data block.
    - batch_size: Batch size.
    - device: Torch device.

    Returns:
    - x, y: Input and target sequences as PyTorch tensors.
    """
    data = get_random_chunk(encode_func, file_path, block_size, batch_size)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
