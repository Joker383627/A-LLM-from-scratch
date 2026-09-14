from collections import Counter
import json

class BPETokenizer:
    """Byte Pair Encoding (BPE) Tokenizer for text processing and compression.

    This class provides functionality to train a BPE tokenizer on a given text file,
    encode arbitrary input text or files into token ID sequences based on learned merge rules,
    and decode token ID sequences back into UTF-8 strings.

    Attributes:
        text (str): The raw input text loaded from a file during initialization.
        tokens (list[int]): Initial list of raw byte values (0-255) extracted from text.
        merge_table (dict[tuple[int, int], int]): Mapping of byte/token pairs to new merged token IDs.
        vocab (dict[int, bytes]): Mapping of token IDs (0 to vocab_size-1) to their corresponding byte sequences.
    """

    def __init__(self, path: str = None, token_number: int = int(5e5)):
        """Initializes the BPETokenizer instance.

        If a file path is provided, reads the text file and converts up to `token_number` 
        characters into initial UTF-8 byte tokens. Otherwise, initializes an empty tokenizer.

        Args:
            path (str, optional): Path to the text file used for training initialization. Defaults to None.
            token_number (int, optional): Maximum number of bytes to read from the file for training. 
                Defaults to 500000.
        """
        if path:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                self.text = f.read()[:token_number]
            utf_bytes = self.text.encode("utf-8")
            self.tokens = list(utf_bytes) if token_number is not None else list(utf_bytes)
        else:
            self.text = ""
            self.tokens = []
            
        self.merge_table = {}
        self.vocab = {ids: bytes([ids]) for ids in range(256)}

    def get_stats(self, tokens: list) -> Counter:
        """Counts frequencies of all adjacent token pairs in a sequence.

        Args:
            tokens (list[int]): A list of integer token IDs.

        Returns:
            Counter: A Counter mapping adjacent token ID pairs `(token_a, token_b)` 
            to their frequency counts in the sequence.
        """
        counts = Counter()
        for pair in zip(tokens[:-1], tokens[1:]):
            counts[pair] += 1
        return counts

    def merge_tokens(self, tokens: list, pair: tuple, new_id: int) -> list:
        """Replaces all non-overlapping occurrences of a specific pair of tokens with a new token ID.

        Args:
            tokens (list[int]): The sequence of token IDs to perform merges on.
            pair (tuple[int, int]): The target pair of adjacent token IDs to be merged.
            new_id (int): The new integer token ID to replace the pair with.

        Returns:
            list[int]: The updated list of token IDs with target pairs replaced by `new_id`.
        """
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def train_BPE(self, vocab_size: int = 400, return_token_length: bool = False):
        """Trains the BPE tokenizer by iteratively merging the most frequent adjacent token pairs.

        Continually scans the token sequence, identifies the most common pair, assigns a new 
        token ID starting from 256, updates the vocabulary and merge tables, and performs 
        the merge across the sequence until reaching `vocab_size` or exhausting pairs.

        Args:
            vocab_size (int, optional): Target total vocabulary size (must be greater than 256). 
                Defaults to 400.
            return_token_length (bool, optional): If True, returns the length of the compressed 
                token sequence alongside merge_table and vocab. Defaults to False.

        Returns:
            tuple: Returns `(self.merge_table, self.vocab, len(tokens))` if `return_token_length` is True,
            otherwise returns `(self.merge_table, self.vocab)`.

        Raises:
            AssertionError: If `vocab_size` is less than or equal to 256.
        """
        assert vocab_size > 256, "Vocab size must be greater than 256"

        tokens = list(self.tokens)
        num_merges = vocab_size - 256

        for k in range(num_merges):
            stats = self.get_stats(tokens)
            if not stats:
                break 

            max_pair = max(stats, key=stats.get)
            idx = 256 + k

            self.merge_table[max_pair] = idx
            self.vocab[idx] = self.vocab[max_pair[0]] + self.vocab[max_pair[1]]
            
            tokens = self.merge_tokens(tokens, max_pair, idx)

        # return (self.merge_table, self.vocab, len(tokens)) if return_token_length else (self.merge_table, self.vocab)

    def encode_text(self, text: str = None, path: str = None, return_text: bool = False, max_bytes: int = int(5e5)):
        """Encodes string input or file contents into a sequence of BPE token IDs.

        Applies learned BPE merge rules stored in `self.merge_table` sequentially to 
        the UTF-8 byte representation of the input text.

        Args:
            text (str, optional): The raw input string to encode. Defaults to None.
            path (str, optional): Path to a text file to read and encode if `text` is None. 
                Defaults to None.
            return_text (bool, optional): If True, returns a tuple `(tokens, text)` containing 
                the encoded IDs and the truncated text string. Defaults to False.
            max_bytes (int, optional): Maximum number of characters to take from the input text 
                prior to encoding. Defaults to 200,000 (`int(2e5)`).

        Returns:
            list[int] or tuple[list[int], str]: The encoded token IDs list, or a tuple containing 
            `(tokens, truncated_text)` if `return_text` is set to True.

        Raises:
            ValueError: If neither `text` nor `path` is provided.
        """
        if text is not None:
            text = text[:max_bytes]
            raw_bytes = text.encode("utf-8", errors="replace")
        elif path is not None:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()[:max_bytes]
            raw_bytes = text.encode("utf-8", errors="replace")
        else:
            raise ValueError("Path or Text must be provided")

        tokens = list(raw_bytes)
        for pair, new_id in self.merge_table.items():
            tokens = self.merge_tokens(tokens, pair, new_id)

        return (tokens, text) if return_text else tokens

    def decode_encoding(self, tokens: list) -> str:
        """Decodes a sequence of BPE token IDs back into a UTF-8 string.

        Maps each token ID back to its original byte sequence via `self.vocab`, concatenates 
        the byte segments, and decodes the result into a human-readable string.

        Args:
            tokens (list[int]): List of integer token IDs to decode.

        Returns:
            str: The reassembled UTF-8 string representation of the token sequence.
        """
        byte_string = b"".join(self.vocab[token_id] for token_id in tokens)
        return byte_string.decode("utf-8", errors="replace")

    def continue_training(self, path: str = None, text: str = None, additional_merges: int = 500, max_chars: int = int(5e5)):
        """Continues BPE training on a new corpus without resetting existing vocabulary.

        First encodes the new corpus using all existing merge rules in `self.merge_table`, 
        then discovers and assigns new token IDs starting after the highest current token ID.

        Args:
            path (str, optional): Path to the new text file for incremental training. Defaults to None.
            text (str, optional): Raw string input for incremental training. Defaults to None.
            additional_merges (int, optional): Number of new merged token IDs to create. Defaults to 500.
            max_chars (int, optional): Maximum characters to load from input text. Defaults to 500,000.

        Raises:
            ValueError: If neither `text` nor `path` is provided.
        """
        if text is not None:
            raw_text = text[:max_chars]
        elif path is not None:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()[:max_chars]
        else:
            raise ValueError("Path or Text must be provided")


        current_tokens = list(raw_text.encode("utf-8", errors="replace"))
        for pair, merge_id in self.merge_table.items():
            current_tokens = self.merge_tokens(current_tokens, pair, merge_id)

        start_idx = max(self.vocab.keys()) + 1

        for k in range(additional_merges):
            stats = self.get_stats(current_tokens)
            if not stats:
                break  

            max_pair = max(stats, key=stats.get)
            idx = start_idx + k


            self.merge_table[max_pair] = idx
            self.vocab[idx] = self.vocab[max_pair[0]] + self.vocab[max_pair[1]]

            current_tokens = self.merge_tokens(current_tokens, max_pair, idx)


    def save(self, path="tokenizer.json"):
        data = {
            "vocab": {
                str(token_id): list(byte_sequence)
                for token_id, byte_sequence in self.vocab.items()
            },

            "merge_table": [
                {
                    "pair": list(pair),
                    "new_id": new_id
                }
                for pair, new_id in self.merge_table.items()
            ]
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)


    @classmethod
    def load(cls, path="tokenizer.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls()

        tokenizer.vocab = {
            int(token_id): bytes(byte_values)
            for token_id, byte_values in data["vocab"].items()
        }

        tokenizer.merge_table = {
            tuple(item["pair"]): item["new_id"]
            for item in data["merge_table"]
        }

        return tokenizer