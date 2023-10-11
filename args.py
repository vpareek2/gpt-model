import argparse

def get_args():
    """
    Parse and return command line arguments as a dictionary.
    """
    parser = argparse.ArgumentParser(description='This is a GPT Language Model.')

    # Adding arguments
    parser.add_argument('-batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('-block_size', type=int, default=128, help='Block size (default: 128)')
    parser.add_argument('-max_iters', type=int, default=200, help='Maximum iterations (default: 200)')
    parser.add_argument('-learning_rate', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
    parser.add_argument('-eval_iters', type=int, default=100, help='Evaluation iterations (default: 100)')
    parser.add_argument('-n_embd', type=int, default=384, help='Embedding size (default: 384)')
    parser.add_argument('-n_head', type=int, default=1, help='Number of heads in attention mechanism (default: 1)')
    parser.add_argument('-n_layer', type=int, default=1, help='Number of layers (default: 1)')
    parser.add_argument('-dropout', type=float, default=0.2, help='Dropout rate (default: 0.2)')

    # Parsing arguments and converting to dictionary
    args = vars(parser.parse_args())
    return args
