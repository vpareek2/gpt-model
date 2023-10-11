def load_chars(file_path):
    """
    Load and return sorted unique characters from a file.

    Args:
    - file_path: Path to the file.

    Returns:
    - chars: Sorted list of unique characters.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        chars = sorted(list(set(text)))
    return chars

def get_mappings(chars):
    """
    Create and return mappings from character to integer and integer to character.

    Args:
    - chars: List of characters.

    Returns:
    - string_to_int: Dictionary mapping characters to integers.
    - int_to_string: Dictionary mapping integers to characters.
    """
    string_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_string = {i: ch for i, ch in enumerate(chars)}
    return string_to_int, int_to_string

def encode(string_to_int, s):
    """
    Convert a string to a list of integers.

    Args:
    - string_to_int: Dictionary for character to integer mapping.
    - s: String to encode.

    Returns:
    - List of integers.
    """
    return [string_to_int[c] for c in s]

def decode(int_to_string, l):
    """
    Convert a list of integers to a string.

    Args:
    - int_to_string: Dictionary for integer to character mapping.
    - l: List to decode.

    Returns:
    - Decoded string.
    """
    return ''.join([int_to_string[i] for i in l])
