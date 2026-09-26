# Tiny LLM

<p align="center">
  <b>A language model built from scratch in PyTorch</b>
</p>

<p align="center">
  Transformer · FineWeb · Distributed Training · GPT-2 Tokenization · CLI Inference
</p>

---

## Overview

**Tiny LLM** is an experimental language-model project built from the ground up to understand the complete pipeline involved in developing a modern autoregressive Transformer language model.

The project is not built by starting from an existing pretrained GPT model. The Transformer architecture, training pipeline, dataset interface, checkpointing system, distributed training setup, and inference pipeline are implemented as part of the project.

The current model is a decoder-only Transformer with approximately **174 million parameters**.

The long-term objective is to take the model through the complete progression:

**Large-scale pretraining → Better language generation → Instruction tuning → Conversational behavior → Physics-domain specialization**

---

## Motivation

Large language models can appear almost magical when viewed only from the outside.

This project is an attempt to remove that abstraction and understand what actually happens underneath:

**Text → Tokenization → Token IDs → Embedding → Self-Attention → Transformer Blocks → Next-token prediction → Loss → Backpropagation → Updated parameters**

Rather than treating a language model as a black-box API, this project investigates the individual components required to build one.

The project also provides an opportunity to experiment with:

- Transformer architectures
- Large-scale text datasets
- Efficient data loading
- GPU training
- Distributed Data Parallelism
- Mixed-precision training
- Gradient accumulation
- Checkpointing
- Text generation
- Instruction tuning
- Domain-specific fine-tuning

---

## Current Model

The current model is a **decoder-only Transformer language model**.

### Model Configuration

| Parameter | Value |
|---|---:|
| Parameters | ~174M |
| Vocabulary size | 50,257 |
| Context length | 768 |
| Embedding dimension | 768 |
| Transformer layers | 8 |
| Attention heads | 8 |
| Feed-forward dimension | 3072 |
| Dropout | 0.1 |
| Tokenizer | GPT-2 / `tiktoken` |
| Architecture | Decoder-only Transformer |

The model uses an autoregressive language-modeling objective.

For a sequence of tokens

`x₁, x₂, ..., xₙ`

the model learns to estimate the probability of the sequence through next-token prediction:

`P(x₁, ..., xₙ) = ∏ᵢ P(xᵢ | x₁, ..., xᵢ₋₁)`

During training, the model receives a sequence of tokens and learns to predict the next token at every position.

---

## Transformer Architecture

The core architecture is implemented in:

`Brain/transformer.py`

The model consists of:

```text
Input token IDs
       │
       ▼
Token Embedding
       │
       +
       │
Positional Embedding
       │
       ▼
┌─────────────────────────┐
│ Transformer Block       │
│                         │
│ LayerNorm               │
│      ↓                  │
│ Causal Self-Attention   │
│      ↓                  │
│ Residual Connection     │
│                         │
│ LayerNorm               │
│      ↓                  │
│ Feed Forward Network    │
│      ↓                  │
│ Residual Connection     │
└─────────────────────────┘
       │
       │ × 8
       ▼
Final LayerNorm
       │
       ▼
Language Model Head
       │
       ▼
Logits
```

The Transformer uses **pre-LayerNorm** blocks.

Each block contains:

1. Layer normalization
2. Multi-head causal self-attention
3. Residual connection
4. Layer normalization
5. Feed-forward network
6. Residual connection

---

## Self-Attention

The model uses multi-head causal self-attention.

For an input representation `X`, the query, key, and value matrices are calculated as:

```text
Q = XWQ
K = XWK
V = XWV
```

The scaled dot-product attention mechanism is:

```text
Attention(Q, K, V)
=
softmax(QKᵀ / √dₖ)V
```

A causal mask prevents each token from attending to tokens appearing later in the sequence.

The implementation uses PyTorch's:

`torch.nn.functional.scaled_dot_product_attention`

where available.

---

## Feed-Forward Network

Each Transformer block contains a position-wise feed-forward network.

The hidden dimension is four times the model embedding dimension:

`768 × 4 = 3072`

The network uses the GELU activation function.

Conceptually:

```text
Input
  │
  ▼
Linear
  │
  ▼
GELU
  │
  ▼
Linear
  │
  ▼
Output
```

---

## Tokenization

The current model uses the GPT-2 tokenizer vocabulary with:

**Vocabulary size = 50,257**

The tokenizer converts raw text into integer token IDs that can be processed by the Transformer.

For example:

```text
"Hello, world!"
```

is transformed into a sequence of token IDs.

The Transformer operates entirely on these token IDs after tokenization.

The project also contains experimental work on implementing BPE tokenization independently:

```text
Tokenizer/
├── tokenizer.py
└── better_bpe_tokenizer.cpp
```

This is part of the broader goal of understanding the complete language-model pipeline rather than treating tokenization as a completely opaque component.

---

## Dataset

The primary pretraining dataset is **FineWeb**.

FineWeb provides a large collection of web text suitable for large-scale language-model pretraining.

The project converts tokenized text into binary files so that the training process can efficiently access sequences without loading the entire dataset into RAM.

The resulting dataset has the general structure:

```text
data/
├── train.bin
├── valid.bin
└── test.bin
```

These files are intentionally excluded from the Git repository because of their size.

---

## Data Pipeline

The complete data pipeline is:

```text
FineWeb
   │
   ▼
Raw text
   │
   ▼
Tokenizer
   │
   ▼
Token IDs
   │
   ▼
Binary files
   │
   ├── train.bin
   ├── valid.bin
   └── test.bin
   │
   ▼
Memory-mapped dataset
   │
   ▼
PyTorch DataLoader
   │
   ▼
Transformer
```

For the current configuration:

**Context length = 768**

A training sample consists of an input sequence and a target sequence shifted by one token:

```text
Input:
x₁ x₂ x₃ ... xₙ

Target:
x₂ x₃ x₄ ... xₙ₊₁
```

The model therefore learns to predict the next token at every position.

---

## Training

The main training implementation is contained in:

```text
Brain/main.py
Brain/train.py
```

The training system supports:

- CUDA acceleration
- Multi-GPU training
- PyTorch Distributed Data Parallel
- Gradient accumulation
- Mixed-precision training
- Automatic checkpointing
- Validation
- Training resumption
- `torch.compile()`

---

## Distributed Training

The project supports multi-GPU training using **PyTorch Distributed Data Parallel (DDP)**.

Current experiments have used two GPUs.

The effective batch size is:

`Batch size per GPU × Number of GPUs × Gradient accumulation steps`

For the current configuration:

`8 × 2 × 4 = 64 sequences/update`

With a context length of 768 tokens:

`64 × 768 = 49,152 tokens/update`

This allows the model to train with a larger effective batch size without requiring the entire batch to fit into GPU memory at once.

---

## Mixed Precision

Training uses mixed-precision computation to reduce GPU memory usage and improve computational efficiency.

The current implementation uses FP16 autocasting together with:

`torch.amp.GradScaler`

Mixed precision allows suitable operations to use reduced precision while maintaining numerical stability during optimization.

---

## Gradient Accumulation

The available GPU memory limits the number of sequences that can be processed in a single forward/backward pass.

Gradient accumulation is therefore used.

Conceptually:

```text
Mini-batch 1
     ↓
forward + backward

Mini-batch 2
     ↓
forward + backward

Mini-batch 3
     ↓
forward + backward

Mini-batch 4
     ↓
forward + backward
     ↓
optimizer.step()
```

The optimizer is updated only after the required number of mini-batches has contributed gradients.

---

## Optimizer

The current training setup uses **AdamW**.

The current configuration includes approximately:

- Learning rate = `3 × 10⁻⁴`
- Weight decay = `0.1`
- Dropout = `0.1`

These values are experimental and may change as training progresses.

---

## Checkpointing

Checkpoint management is implemented in:

`Brain/checkpoints.py`

A training checkpoint stores information required to continue training, including:

- Model parameters
- Optimizer state
- Gradient scaler state
- Epoch
- Batch position
- Best validation loss

This allows training to be interrupted and resumed without restarting from randomly initialized parameters.

Model checkpoints are intentionally excluded from Git.

---

## Inference

The inference system is currently split between:

```text
GPTapp/generate.py
chat.py
```

The model can be loaded from a trained checkpoint and used interactively from the terminal.

A basic session looks like:

```text
=================== WELCOME ====================

please set your name :
Name : Joker

Joker : Hello

GPT : ...
```

The current interface is intentionally minimal.

The purpose at this stage is to establish a complete path from:

**Checkpoint → Model → Tokenizer → Generation → Interactive conversation**

---

## Command-Line Interface

The current chat application can be launched using the checkpoint supplied as a command-line argument.

For example:

```bash
python3 chat.py latest_checkpoint_3.pt
```

The CLI currently provides:

- Model loading
- Checkpoint loading
- User-name configuration
- Interactive input
- Model generation
- Basic exit handling

The interface is a foundation for a more complete local AI application.

---

## Current Generation System

The generation system is still under active development.

The current implementation relies heavily on a maximum generation length.

The next generation system will introduce more sophisticated sampling and stopping behavior.

Planned improvements include:

- Temperature sampling
- Top-k sampling
- Top-p / nucleus sampling
- EOS token detection
- Better stopping criteria
- Context-window management
- Streaming token generation
- Conversation history management

One particularly important objective is allowing the model to learn **when a response should naturally end**, rather than relying entirely on a hard generation limit.

---

## Why the Current Model Is Not Yet a Chatbot

Although the model can generate coherent English text, the current model should not yet be considered a conversational assistant.

The reason is primarily the training objective.

During pretraining, the model is learning:

> Predict the next token.

It is not explicitly learning:

```text
USER
    ↓
ASSISTANT
    ↓
EOS
```

As a result, the model may:

- Continue generating for too long
- Change topics unexpectedly
- Produce incomplete fragments
- Repeat common textual patterns
- Generate text that resembles dialogue without maintaining a consistent conversation
- Fail to determine when an answer should end

These behaviors are expected for a base language model trained primarily on general web text.

Instruction tuning is intended to address this later in the project.

---

## Current Model Behavior

The current model has demonstrated the ability to generate coherent English continuations and conversational-looking text.

However, the model still frequently behaves like a text continuation system rather than a true assistant.

This is an important distinction in the development of the project.

The goal is not to artificially hide these limitations, but to use them as a way to understand how different stages of language-model training affect behavior.

---

# Development Roadmap

## Stage 1 — Transformer and Training Infrastructure

- [x] Implement Transformer architecture
- [x] Implement multi-head self-attention
- [x] Implement causal attention
- [x] Implement Transformer blocks
- [x] Implement language-model head
- [x] Implement tokenized binary datasets
- [x] Implement training loop
- [x] Implement validation
- [x] Implement checkpointing
- [x] Implement gradient accumulation
- [x] Implement mixed precision
- [x] Implement multi-GPU DDP
- [x] Train a ~174M parameter model
- [x] Build a basic CLI inference system

---

## Stage 2 — Improved Pretraining

The next stage is to continue training the existing model using a substantially larger amount of FineWeb data.

Instead of restarting from random initialization, the existing checkpoint will be loaded and training will continue:

```text
Existing 174M model
        │
        ▼
Load checkpoint
        │
        ▼
More FineWeb training
        │
        ▼
Improved language model
```

This stage will focus on:

- Larger training corpus
- Longer pretraining
- Improved generation
- Validation loss tracking
- Checkpoint comparison
- Qualitative generation tests

---

## Stage 3 — Evaluation

A fixed collection of prompts will be used to compare different model checkpoints.

The evaluation will include categories such as:

- General language
- Explanation
- Short-form writing
- Long-form continuation
- Mathematical text
- Scientific text
- Physics questions

The purpose is to compare checkpoints systematically rather than relying entirely on individual conversations.

---

## Stage 4 — Instruction Fine-Tuning

Once the base language model is sufficiently capable, the next major stage is instruction tuning.

The training format will move toward structured conversations such as:

```text
User:
Explain Newton's second law.

Assistant:
Newton's second law states that ...
```

rather than arbitrary web-text continuation.

The goal is to teach the model the structure of useful interactions:

```text
USER
  ↓
ASSISTANT
  ↓
EOS
```

This stage is intended to improve:

- Instruction following
- Conversational consistency
- Response formatting
- Response termination
- Question answering
- Task-oriented generation

---

# Physics Specialization

One of the long-term goals of Tiny LLM is to specialize the model for physics.

The intended progression is:

```text
General language model
        ↓
Instruction-tuned language model
        ↓
Physics-domain training
        ↓
Physics-focused language model
```

Potential areas include:

- Classical Mechanics
- Quantum Mechanics
- Electrodynamics
- Statistical Mechanics
- Special Relativity
- General Relativity
- Quantum Field Theory
- Particle Physics
- High Energy Physics
- Mathematical Physics

Possible domain-specific data sources include:

- Physics textbooks
- Lecture notes
- Scientific papers
- Problem-solving datasets
- Mathematical physics material
- Carefully constructed physics instruction-response pairs

The objective is not merely to make the model produce physics terminology.

A more interesting goal is to investigate how much useful domain knowledge and problem-solving behavior can be obtained from targeted training of a relatively small language model.

---

# Why Build a Small Model?

The current model is small compared with modern production-scale language models.

A model with approximately 174M parameters is nowhere near the scale of systems containing billions of parameters.

That limitation is intentional.

A smaller model makes it possible to experiment with:

- Architecture
- Dataset composition
- Tokenization
- Optimization
- Training schedules
- Fine-tuning
- Sampling strategies
- Domain specialization

without requiring enormous computational resources.

The purpose of this project is therefore not to compete with large commercial language models.

The purpose is to understand the engineering and scientific ideas behind them.

---

# Project Structure

```text
Tiny-LLM/
│
├── Brain/
│   ├── __init__.py
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
│   ├── __init__.py
│   ├── tokenizer.py
│   └── better_bpe_tokenizer.cpp
│
├── chat.py
├── config.py
├── LLM.ipynb
├── test.ipynb
├── .gitignore
└── README.md
```

### `Brain/`

Contains the core model and training implementation.

### `transformer.py`

Defines:

- Multi-head attention
- Transformer blocks
- Transformer model

### `train.py`

Contains training and evaluation routines.

### `checkpoints.py`

Handles saving and loading training checkpoints.

### `preprocess.py`

Contains dataset implementations used to access tokenized training data.

### `main.py`

Acts as the primary training entry point and handles the overall training setup, including distributed training.

### `GPTapp/`

Contains inference and generation functionality.

### `Tokenizer/`

Contains tokenizer-related implementations and experiments.

### `chat.py`

Provides the command-line interface for interacting with the trained model.

### `config.py`

Contains model and training configuration.

---

# Reproducibility

The project is designed so that the major components of the training pipeline can be reproduced independently.

At minimum, reproduction requires:

- Python
- PyTorch
- CUDA-compatible GPU(s)
- Tokenized training data

The exact training configuration is defined in the source code.

Because the training dataset and model checkpoints are large, they are intentionally excluded from this Git repository.

---

# Hardware and Efficiency

Training a Transformer of this size is computationally demanding.

The project has been developed under relatively limited hardware resources and therefore makes extensive use of:

- Mixed precision
- Gradient accumulation
- Memory-mapped datasets
- Distributed Data Parallelism
- Efficient attention implementations
- Checkpoint resumption
- GPU acceleration

These constraints make memory efficiency an important part of the project.

The model is therefore not only an experiment in machine learning, but also an experiment in practical ML engineering under limited computational resources.

---

# Design Philosophy

The central idea behind Tiny LLM is:

> **Build the system yourself, understand every major component, and progressively improve it.**

The project follows a progression from fundamental components to increasingly sophisticated behavior:

```text
Tokenizer
    ↓
Dataset
    ↓
Transformer
    ↓
Pretraining
    ↓
Generation
    ↓
Instruction tuning
    ↓
Conversation
    ↓
Physics specialization
```

Each stage introduces a different technical problem.

### Pretraining

How does a model acquire statistical knowledge of language?

### Generation

How should probability distributions over tokens be converted into text?

### EOS

How does a model learn when a response is finished?

### Instruction tuning

How does next-token prediction become useful instruction following?

### Domain specialization

How much can targeted data change the behavior of a relatively small language model?

These questions are central to the development of the project.

---

# Future Directions

Once the basic conversational model is working, the project can be extended beyond simple text generation.

Possible future directions include:

- Retrieval-Augmented Generation
- External tool use
- Python execution
- Mathematical tools
- Physics calculation tools
- Local document retrieval
- Conversation history
- Structured reasoning experiments
- Physics problem-solving datasets
- Domain-specific evaluation

The long-term vision is a **small, local, physics-oriented language model** whose architecture and training pipeline are understood from the ground up.

---

# Current Status

**Current stage: Base language model + early CLI inference**

The project currently contains:

- A working ~174M parameter decoder-only Transformer
- GPT-2-compatible vocabulary
- FineWeb-based pretraining
- Binary memory-mapped datasets
- Multi-GPU DDP training
- Gradient accumulation
- Mixed precision
- Checkpoint saving and loading
- Basic text generation
- A functional command-line interface

The current model can generate coherent English text, but it has not yet undergone dedicated instruction tuning.

The immediate development priorities are:

```text
1. Improve generation
2. Continue pretraining
3. Evaluate the improved model
4. Build instruction datasets
5. Fine-tune for conversation
6. Specialize the model for physics
```

---

# Acknowledgements

This project builds on the broader ecosystem of open-source machine-learning research and software, particularly:

- PyTorch
- FineWeb
- Hugging Face Datasets
- GPT-2 tokenizer / `tiktoken`
- CUDA and GPU acceleration technologies
- Videos by Andrej Karapthy (nanoGPT)
- Building-LLM-from-Scratch,Sebastian Raschka

The project is intended as an independent educational and experimental implementation built using these tools and resources.

---

# Author

**Tuhin Bakuli**

MSc Physics  
Indian Institute of Technology Kanpur

---
