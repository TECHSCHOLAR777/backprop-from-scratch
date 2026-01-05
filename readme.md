I built this project to understand how forward propagation, loss computation, and gradient backpropagation actually work internally, instead of relying on high-level libraries.

What this project supports

Multi-layer neural networks

ReLU and Sigmoid activation functions

Binary Cross-Entropy loss (with Sigmoid output layer)

Manual implementation of gradient computation using the chain rule

Why I built this

I am a first-year undergraduate student, and this project is part of my learning process to deeply understand:

how gradients flow backward through a network

why backpropagation is efficient (O(N + P))

how neural networks learn without using frameworks like PyTorch or TensorFlow

This implementation is educational, not production-ready, and focuses on clarity and correctness of logic.
