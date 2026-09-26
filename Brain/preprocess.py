# from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from Tokenizer.tokenizer import BPETokenizer


class FastBinaryTextDataSet(Dataset):
    def __init__(self, path, context_length, stride):
        self.context_length = context_length
        self.stride = stride
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        N, C, S = len(self.data), context_length, stride
        self.n_sample = max(0, (N - C - 1) // S + 1)

    def __len__(self):
        return self.n_sample

    def __getitem__(self, index):
        start = index * self.stride
        end = start + self.context_length
        x = torch.from_numpy(self.data[start:end].astype(np.int64, copy=False))
        y = torch.from_numpy(self.data[start+1:end+1].astype(np.int64, copy=False))
        return x, y


# TOKENIZER_PATH = Path(__file__).resolve().parent.parent /"Tokenizer" / "new_tok.json"

# tokenizer = BPETokenizer.load(TOKENIZER_PATH)
# # tokenizer = BPETokenizer.load("/home/tuhin/python_codes/Tiny LLM/Tokenizer/new_tok.json")

# print(TOKENIZER_PATH)

# class TextDataset(Dataset):

#     def __init__(self,text:str = None,path:str = None,context_length:int = 256,stride:int = 1,tokenizer = tokenizer):
#         super().__init__()

#         if text is None:
#             if path is not None:
#                 with open(path) as f:
#                     self.text = f.read()
#             else:
#                 raise ValueError("Path or Text must be provided")
#         else:
#             self.text = text

#         self.tokenizer = tokenizer
#         self.encoding = torch.tensor(self.tokenizer.encode(self.text,None,False,len(self.text)))

#         self.input_id = self.encoding[:-1].unfold(0,context_length,stride)
#         self.target_id = self.encoding[1:].unfold(0,context_length,stride)

#     def __len__(self):
#         return len(self.input_id)

#     def __getitem__(self, idx):
#         return self.input_id[idx],self.target_id[idx]


# class BinaryTextDataSet(Dataset):
#     """This Dataset expects a Binary file of pretokenized data"""
#     def __init__(self,path,context_length,stride):
#         super().__init__()
#         self.context_length = context_length
#         self.stride = stride

#         self.data = np.memmap(path,dtype = np.uint16,mode = "r")

#         self.n_sample = (len(self.data) - context_length + stride - 1)//stride

#     def __len__(self):
#         return self.n_sample

#     def __getitem__(self, index):
#         start = index * self.stride
#         end = start + self.context_length

#         input_id = torch.tensor(self.data[start:end].copy()).long()
#         target_id = torch.tensor(self.data[start+1 : end+1].copy()).long()

#         return input_id,target_id


# def prepare_binary_data(read_file_path, data_chunk,tokenizer):
#     with open(read_file_path,"r",encoding="utf-8") as fin:
#         with open("train.bin","wb") as train_bin, open("test.bin","wb") as test_bin, open("valid.bin","wb") as valid_bin:
#             i = 0
#             while True:
#                 text = fin.read(data_chunk)

#                 if not text:
#                     break

#                 tokens = tokenizer.encode(text)
#                 arr = np.asarray(tokens,dtype=np.uint16)
#                 if i < 400 :
#                     arr.tofile(valid_bin)
#                     i += 100
#                 elif i < 600:
#                     arr.tofile(test_bin)
#                     i += 100
#                 else:
#                     arr.tofile(train_bin)
    
