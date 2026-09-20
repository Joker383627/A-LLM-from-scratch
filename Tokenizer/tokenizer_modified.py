from collections import Counter
import json
from .bpe_encoder import BPEEncoder
from .bpe_trainer import BPETrainer

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

    def __init__(self):
        """Initializes the BPETokenizer instance.

        If a file path is provided, reads the text file and converts up to `token_number` 
        characters into initial UTF-8 byte tokens. Otherwise, initializes an empty tokenizer.

        Args:
            path (str, optional): Path to the text file used for training initialization. Defaults to None.
            token_number (int, optional): Maximum number of bytes to read from the file for training. 
                Defaults to 500000.
        """ 
        self.merge_table = {}
        self.vocab = {}
        self.cpp_encoder = None
        self.cpp_trainer = None


    def train_BPE(self,path:str = None,text:str = None,
                   vocab_size: int = 400,start_id: int = 256, 
                   return_token_length: bool = False):
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

        if text is None:
            if path is not None:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            else:
                raise ValueError("Path or Text must be provided")

        raw_bytes = text.encode("utf-8", errors="replace")

        raw_bytes = text.encode("utf-8", errors="replace")
        self.tokens = list(raw_bytes)

        tokens = list(self.tokens)


        self.cpp_trainer = BPETrainer()
        tokens = self.cpp_trainer.train_bpe(tokens,vocab_size,start_id)

        self.cpp_vocab = self.cpp_trainer.get_vocab()
        self.cpp_merge_table = self.cpp_trainer.get_merge_table()

        self.vocab = {
            int(token_id):bytes(byte_values)
            for token_id,byte_values in self.cpp_vocab.items()
        }

        self.merge_table = {
            tuple(pair): int(new_id)
            for pair,new_id in self.cpp_merge_table.items()
        }

        self.tokens = tokens
        if return_token_length:
            return len(tokens)

    def build_cpp_encoder(self):
        pairs = []
        ids = []

        self.merge_table = dict(sorted(self.merge_table.items(),key = lambda x: x[1]))

        for pair, new_id in self.merge_table.items():
            pairs.append(pair)
            ids.append(new_id)

        self.cpp_encoder = BPEEncoder(pairs, ids)

    def encode(self, text: str = None, path: str = None, 
                    return_text: bool = False, 
                    max_bytes: int = int(5e5)):
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
        if self.cpp_encoder is None:
            self.build_cpp_encoder()

        tokens = self.cpp_encoder.encode(text)

        return (tokens, text) if return_text else tokens

    def decode(self, tokens: list) -> str:
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

    def continue_training(self, path: str = None, text: str = None, 
                          vocab_size : int = None, 
                          max_chars: int = int(5e5)):
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
        if text is None:
            if path is not None:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            else:
                raise ValueError("Path or Text must be provided")

        current_tokens = list(text.encode("utf-8", errors="replace"))

        # tokens = list(current_tokens)

        self.cpp_trainer.set_vocab(self.cpp_vocab)
        self.cpp_trainer.set_merge_table(self.cpp_merge_table)

        for pair, merge_id in self.merge_table.items():
            current_tokens = self.cpp_trainer.merge_token(current_tokens, pair, merge_id)

        start_id = max(self.vocab.keys()) + 1

        tokens = self.cpp_trainer.train_bpe(current_tokens,vocab_size,start_id)

        self.cpp_vocab = self.cpp_trainer.get_vocab()
        self.cpp_merge_table = self.cpp_trainer.get_merge_table()

        self.vocab = {
            int(token_id):bytes(byte_values)
            for token_id,byte_values in self.cpp_vocab.items()
        }

        self.merge_table = {
            tuple(pair): int(new_id)
            for pair,new_id in self.cpp_merge_table.items()
        }



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
