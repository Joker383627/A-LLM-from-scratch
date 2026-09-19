from pathlib import Path
import torch
from torch.utils.data import Dataset
from Tokenizer.tokenizer import BPETokenizer

TOKENIZER_PATH = Path(__file__).resolve().parent.parent /"Tokenizer" / "new_tok.json"

tokenizer = BPETokenizer.load(TOKENIZER_PATH)
# tokenizer = BPETokenizer.load("/home/tuhin/python_codes/Tiny LLM/Tokenizer/new_tok.json")

print(TOKENIZER_PATH)

class TextDataset(Dataset):

    def __init__(self,text:str = None,path:str = None,context_length:int = 256,stride:int = 1,tokenizer = tokenizer):
        super().__init__()

        if text is None:
            if path is not None:
                with open(path) as f:
                    self.text = f.read()
            else:
                raise ValueError("Path or Text must be provided")
        else:
            self.text = text

        self.tokenizer = tokenizer
        self.encoding = torch.tensor(self.tokenizer.encode_text(self.text,None,False,len(self.text)))

        self.input_id = self.encoding[:-1].unfold(0,context_length,stride)
        self.target_id = self.encoding[1:].unfold(0,context_length,stride)

    def __len__(self):
        return len(self.input_id)

    def __getitem__(self, idx):
        return self.input_id[idx],self.target_id[idx]


    
