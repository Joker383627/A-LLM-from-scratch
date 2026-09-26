# from Brain.transformer import Transformer

import torch
# import tiktoken
device = "cuda" if torch.cuda.is_available() else "cpu"

# tokenizer = tiktoken.get_encoding("gpt2")
from config import *

@torch.no_grad()
def generate(model,prompt, tokenizer, max_new_tokens=100, temperature=1.0):
    model.eval()

    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):

        # Keep only the most recent context
        x_cond = x[:, -CONTEXT_LENGTH:]
        # print("printed : ",x_cond)
        logits = model(x_cond)

        # Take logits for the final position
        logits = logits[:, -1, :] / temperature

        probs = torch.softmax(logits, dim=-1)

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)

        x = torch.cat((x, next_token), dim=1)

    generated_tokens = x[0].tolist()

    return tokenizer.decode(generated_tokens)[len(prompt):]