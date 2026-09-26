print("importing dependencies...")
import sys
from pathlib import Path
from Brain.transformer import Transformer
from GPTapp.generate import generate
import torch
import tiktoken
from config import *

print("importing done")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = tiktoken.get_encoding("gpt2")
print("tokenizer initialized...\n")

print("preparing model ...")
model = Transformer(
        emb_dim=EMBEDDED_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        num_layers=NUM_LAYERS,
        is_causal=True,
        context_length=CONTEXT_LENGTH,
        vocab_size=VOCAB_SIZE
    ).to(device)
print("model ready, loading parameters...")

checkpoint_name = sys.argv[1]

param_path = Path(checkpoint_name).resolve().parent/"checkpoints"/checkpoint_name
state_dict = torch.load(param_path,map_location=device)
model.load_state_dict(state_dict["model"])

print("loading done...")

print("starting tinyGPT...")
print(f"=================== WELCOME ====================")
print("please set your name : ")
name = input("Name : ")
while True:
    prompt = input(name + str(" : "))
    if prompt in ["Quit","quit","exit","q"]:
        print("Do you wanna get out? (Y/N)")
        answer = input("answer : " )
        if answer in ["Y","y"]:
            break
        else:
            print("Continuing to app...")
            continue
    answer = generate(model = model,tokenizer=tokenizer,prompt=prompt,max_new_tokens=40,temperature=0.8)
    print(f"GPT : {answer}")