# Tiny LLM

A language model built from scratch in PyTorch, including the Transformer architecture, training pipeline, tokenizer, inference system, and a simple command-line chat interface.

The goal of this project is to understand and build a modern language model from the ground up rather than relying on a pretrained model or framework abstraction.

---

## Current Model

The current model is a decoder-only Transformer with approximately **174M parameters**.

### Architecture

| Component | Value |
|---|---:|
| Vocabulary size | 50,257 |
| Context length | 768 tokens |
| Embedding dimension | 768 |
| Transformer layers | 8 |
| Attention heads | 8 |
| Feed-forward dimension | 3072 |
| Dropout | 0.1 |
| Architecture | Decoder-only Transformer |
| Tokenizer | GPT-2 / `tiktoken` |
| Parameters | ~174M |

The Transformer uses:

- Multi-head self-attention
- Causal attention
- Pre-LayerNorm Transformer blocks
- GELU activation
- Learned token embeddings
- Learned positional embeddings
- Final LayerNorm
- Language-model output head

---

## Project Structure

```text
Tiny-LLM/
│
├── Brain/
│   ├── checkpoints.py
│   ├── main.py
│   ├── preprocess.py
│   ├── train.py
│   └── transformer.py
│
├── GPTapp/
│   └── generate.py
│
├── Tokenizer/
│   ├── tokenizer.py
│   └── better_bpe_tokenizer.cpp
│
├── chat.py
├── config.py
├── LLM.ipynb
├── test.ipynb
├── .gitignore
└── README.md
