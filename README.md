# Understanding LLM Architecture: Encoder, Decoder, Self-Attention and Multi-Head Attention

Modern Large Language Models (LLMs) such as GPT, BERT, and T5 are built on the Transformer architecture, introduced by Vaswani et al. in the 2017 paper "Attention is All You Need". This architecture marked a turning point in natural language processing by replacing sequential models like RNNs with a parallelized, attention-based approach. 

This document provides a detailed explanation of the key components that form the foundation of LLMs, with the goal of helping developers and researchers understand how these models function under the hood.

## Encoder

The encoder is responsible for processing the input sequence and producing a contextualized representation of each token. Unlike recurrent architectures, it handles the entire input simultaneously, allowing for faster training and better handling of long-range dependencies.

Each token is compared with all others in the sequence using self-attention, enabling the encoder to model the relationships between words regardless of their distance. This is particularly useful in tasks where the structure and meaning of the entire input must be captured, such as sentiment analysis, classification, or entity recognition.

BERT is a prime example of an encoder-only architecture. It reads full input sequences bidirectionally and is designed for understanding and encoding text.

## Decoder

The decoder is used to generate output sequences, such as text completions or translations. It operates autoregressively, producing one token at a time and using previously generated tokens as context for the next prediction.

In standalone decoder architectures like GPT, the model learns to predict the next word in a sentence by focusing only on past context. This makes it highly effective for tasks such as text generation, code completion, and conversational AI.

The decoder also incorporates attention over encoder outputs in encoder-decoder setups, allowing it to focus on relevant parts of the input while generating output.

## Encoder-Decoder Architecture

This architecture combines both encoders and decoders into a unified framework. The encoder processes the input and passes a contextual representation to the decoder, which then generates the output.

This structure is especially suited for tasks like machine translation, summarization, and question answering, where the input and output formats differ. The encoder extracts the meaning of the input, and the decoder constructs a coherent, context-aware output based on that understanding.

T5 and MarianMT are prominent examples of encoder-decoder models that handle text-to-text tasks with flexibility and precision.

## Self-Attention

Self-attention is a mechanism that enables the model to weigh the importance of different tokens in a sequence when encoding or generating a particular token. Every token in the input creates a query, a key, and a value vector, which are used to calculate attention scores.

These scores determine how much focus the model should place on each token when building its internal representation. The attention mechanism allows the model to capture semantic relationships, such as subject-object dependencies or contextual clues spread across the sequence.

This attention-driven representation is the core innovation that allows transformers to outperform previous models in understanding context and relationships.

## Multi-Head Attention

Multi-head attention extends the self-attention mechanism by running multiple attention operations in parallel. Each head learns to focus on different aspects of the sequence, such as syntax, semantics, or position.

The outputs of these heads are concatenated and linearly transformed, resulting in a rich, multifaceted representation. This diversity enhances the model’s ability to capture complex relationships within the data.

Multi-head attention is used in both encoder and decoder layers, enabling robust information flow across different parts of the network.

## Comparison: RNNs vs Transformers

While Recurrent Neural Networks process sequences step-by-step and rely heavily on the order of tokens, Transformers process the entire input at once. This parallelism not only speeds up training but also allows for better modeling of long-distance dependencies.

Transformers have largely replaced RNNs in state-of-the-art NLP systems due to their superior scalability, efficiency, and context modeling capabilities.

| Feature               | RNN                                | Transformer                        |
|-----------------------|-------------------------------------|-------------------------------------|
| Processing            | Sequential                          | Parallel                             |
| Context Range         | Limited (short memory)              | Global (via attention)               |
| Training Speed        | Slow                                | Fast                                 |
| Scalability           | Poor                                | Excellent                            |
| Use in LLMs           | Rare in modern architectures        | Core of all modern LLMs             |

## Applications of LLM Architectures

Text classification  
Summarization  
Machine translation  
Question answering  
Text and code generation  
Conversational agents  
Information retrieval (RAG)  
Multimodal tasks (text, image, audio fusion)

## Learn More

To explore these concepts further and dive into practical examples, visit the following resources:

Paper: Attention Is All You Need: https://arxiv.org/abs/1706.03762  
Visual Guide: The Illustrated Transformer by Jay Alammar: https://jalammar.github.io/illustrated-transformer/  
Model Hub: https://huggingface.co/models  
Project Repository: https://github.com/donaldtagne/Transformer-Architecture

## About This Document

This overview was created to serve as a reference for developers, students, and researchers who are looking to build a deeper understanding of transformer-based models. It can be used as a standalone explanation or linked from posts, tutorials, or documentation.
