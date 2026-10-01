# Neural Network

My first neural network, built from scratch without libraries like `pytorch` or `tensorflow` to recognize handwritten digits.

It takes a **28×28 grayscale image** as input and predicts which digit from `0–9` was drawn.

The network has two hidden layers with **64 and 32 neurons**, and was trained on the MNIST dataset. The current saved weights reach around **90% accuracy**.

## How it works

```text
28 × 28 image
      ↓
   784 inputs
      ↓
  64 neurons
      ↓
  32 neurons
      ↓
  10 outputs
      ↓
 Predicted digit
```

The network and training process are implemented manually rather than using a machine-learning framework.

## Files

```text
Neural-network/
├── making ai.py             # Creates and trains the network
├── using ai.py              # Draw a digit and get a prediction
├── quantize.py              # Converts image values to the required format
└── mnist_model_weights.npz  # Trained model weights
```

`using ai.py` provides a small **28×28 drawing window** where you can draw a digit and have the network classify it.

## Dataset

The network is trained using the **MNIST handwritten digit dataset**.

The training data used by the project can be obtained from Kaggle.

## Why I made it

This was one of my first projects where I actually implemented and trained a neural network myself, rather than just using a machine-learning library.

It was mainly a way to understand what is happening behind the usual `model.fit()` approach.
