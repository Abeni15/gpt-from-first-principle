# Building a GPT from First Principles

A decoder-only GPT Transformer trained on the Little Shakespeare dataset, built from scratch in PyTorch.  

---

## Overview

This project implements a GPT-style language model entirely from scratch — no HuggingFace, no pre-built Transformer libraries. Every component (self-attention, FFN, residuals, LR scheduling, etc.) is hand-coded to demonstrate a deep understanding of the architecture.

The model is trained on tinyshakespeare (~1 MB of Shakespeare text) using character-level tokenization, and generates new Shakespeare-style text autoregressively after training.

---

## Project Structure

├── gpt_shakespeare.ipynb   # Main Colab notebook (full implementation)
├── README.md               # This file
---

## Architecture

| Component | Detail |
|---|---|
| Model type | Decoder-only GPT |
| Tokenization | Character-level (vocab = 65) |
| Context length | 128 characters |
| Embedding dim | 256 |
| Attention heads | 8 (d_k = 32 per head) |
| Decoder layers | 6 |
| FFN hidden dim | 1024 (4 × d_model) |
| Activation | GELU |
| Total parameters | ~5.7M |

---

## Training Setup

| Setting | Value |
|---|---|
| Optimizer | AdamW (weight decay = 0.1) |
| Learning rate | 3e-4 (peak) |
| LR schedule | Linear warmup (200 steps) + cosine decay |
| Gradient clipping | 1.0 |
| Batch size | 64 |
| Training steps | 5000 |
| Train / Val split | 90% / 10% |
| Checkpointing | Every 500 steps (best val loss) |

---

## How to Run

1. Open gpt_shakespeare.ipynb in [Google Colab](https://colab.research.google.com/)
2. Set runtime to GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells top to bottom
4. Training takes ~10–15 minutes on a T4 GPU
5. Generated text and loss curves are produced at the end

---

## Key Concepts Implemented

- Token + Positional Embeddings — x = tok_emb + pos_emb
- Multi-Head Causal Self-Attention — Q, K, V projections with causal mask (future = −∞)
- Feed-Forward Network — FFN(x) = W2 · GELU(W1·x + b1) + b2
- Residual Connections + LayerNorm — x + sublayer(LayerNorm(x))
- Decoder Stack — 6 identical blocks stacked
- Cross-Entropy Loss — next-token prediction objective
- AdamW + Warmup + Cosine Decay — full training loop
- Gradient Clipping — prevents exploding gradients
- Autoregressive Generation — temperature + top-k sampling

---

## Sample Output

After training, the model generates text like:

KING RICHARD:
What means this, my lord? the king is gone,
And all the world is full of sorrow now,
That we must weep for what we cannot keep.
---

## Trainer

Saad Musema — saadmusema3@gmail.com
