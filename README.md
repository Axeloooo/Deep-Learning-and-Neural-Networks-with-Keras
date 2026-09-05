# Deep Learning and Neural Networks with Keras

[![Release](https://github.com/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras/actions/workflows/release.yaml/badge.svg)](https://github.com/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras/actions/workflows/release.yaml)
[![GitHub Tag](https://img.shields.io/github/v/tag/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras)](https://github.com/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras/tags)
[![GitHub Issues](https://img.shields.io/github/issues/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras)](https://github.com/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras)](https://github.com/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras/pulls)
[![Contributors](https://img.shields.io/github/contributors/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras)](https://github.com/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras/graphs/contributors)
[![Repo Size](https://img.shields.io/github/repo-size/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras)](https://github.com/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Conventional Commits](https://img.shields.io/badge/commits-Conventional%20Commits-fa6673)](https://www.conventionalcommits.org/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras/blob/main/notebooks/forward-propagation.ipynb)

An educational collection of eight Jupyter notebooks exploring neural networks, from a forward pass written in NumPy to Keras models for regression, image classification, and sequence modeling. Google Colab is the supported environment for running the lessons.

The collection is a work in progress. The notebooks contain explanations, worked examples, and practice exercises; they will continue to be refined, including the lessons on backpropagation and activation functions.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started with Google Colab](#getting-started-with-google-colab)
- [Notebooks and Learning Path](#notebooks-and-learning-path)
- [Core Concepts](#core-concepts)
- [Data and Runtime Notes](#data-and-runtime-notes)
- [Contributing](#contributing)
- [Conclusion](#conclusion)
- [References](#references)
- [License](#license)

## Introduction

A neural network combines weighted sums with nonlinear functions to transform inputs into predictions. Training adjusts its weights and biases to reduce a loss: a measure of how far those predictions are from the desired outputs.

These lessons connect the underlying arithmetic to higher-level Keras APIs. Start by tracing values through a small network, then study how gradients guide learning. Continue with supervised learning on tabular data and handwritten digits, before exploring convolution, attention, and pretrained models.

Basic Python, NumPy arrays, and matrix multiplication are useful prerequisites. Familiarity with derivatives and the chain rule will help with backpropagation. Each notebook is a separate lesson with its own setup cells; the order below provides a conceptual progression.

## Getting Started with Google Colab

1. Open a notebook using its **Open in Colab** link below and sign in to your Google account.
2. Save a copy in Google Drive if you want to keep your edits and exercise answers.
3. Connect to a Python runtime. A CPU runtime is sufficient for the NumPy lessons; GPU acceleration can help with the image-model lessons when available.
4. Read and execute the cells from top to bottom. Later cells depend on variables and models created earlier.
5. Try the practice exercises, compare the results, and save your copy before ending the session.

There is no local installation procedure or dependency lockfile in this repository. The notebooks rely on the Colab environment; leave the commented `!pip install` cells commented out. Some lesson text still refers to the original Skills Network environment. If an import fails in Colab, record the error and installed library versions in an issue so compatibility can be investigated.

Colab runtimes are temporary, and accelerator availability and usage limits vary. Download any generated files you want to keep. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html) for runtime and storage details.

## Notebooks and Learning Path

| Order | Notebook | What you will explore | Run |
| --- | --- | --- | --- |
| 1 | [Forward propagation](notebooks/forward-propagation.ipynb) | Weights, biases, sigmoid activation, and a forward pass with NumPy | [Open in Colab](https://colab.research.google.com/github/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras/blob/main/notebooks/forward-propagation.ipynb) |
| 2 | [Backpropagation](notebooks/backward-propagation.ipynb) | Training a small NumPy network on XOR and tracking error | [Open in Colab](https://colab.research.google.com/github/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras/blob/main/notebooks/backward-propagation.ipynb) |
| 3 | [Activation functions and vanishing gradients](notebooks/activation-functions-and-vanishing.ipynb) | Activation functions and their derivatives using NumPy and plots | [Open in Colab](https://colab.research.google.com/github/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras/blob/main/notebooks/activation-functions-and-vanishing.ipynb) |
| 4 | [Regression with Keras](notebooks/regression-with-keras.ipynb) | Predicting concrete compressive strength from tabular inputs | [Open in Colab](https://colab.research.google.com/github/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras/blob/main/notebooks/regression-with-keras.ipynb) |
| 5 | [Classification with Keras](notebooks/classification-with-keras.ipynb) | Dense networks for MNIST handwritten-digit classification | [Open in Colab](https://colab.research.google.com/github/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras/blob/main/notebooks/classification-with-keras.ipynb) |
| 6 | [Convolutional neural networks](notebooks/convolutional-neural-networks-with-keras.ipynb) | Convolution and pooling for MNIST image classification | [Open in Colab](https://colab.research.google.com/github/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras/blob/main/notebooks/convolutional-neural-networks-with-keras.ipynb) |
| 7 | [Sequence modeling with attention](notebooks/transformers-with-Keras.ipynb) | An LSTM encoder–decoder with attention on five English–Spanish sentence pairs | [Open in Colab](https://colab.research.google.com/github/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras/blob/main/notebooks/transformers-with-Keras.ipynb) |
| 8 | [Classification and captioning](notebooks/classification-and-captioning.ipynb) | Aircraft-damage classification with VGG16 and image captioning with BLIP | [Open in Colab](https://colab.research.google.com/github/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras/blob/main/notebooks/classification-and-captioning.ipynb) |

Despite its filename, `transformers-with-Keras.ipynb` currently builds an LSTM encoder–decoder whose decoder attends to encoder outputs. It is an attention lesson, not an implementation of a full Transformer architecture.

## Core Concepts

### Forward propagation

A layer computes a weighted sum plus a bias, then applies an activation function. For a column-vector input, this can be written as `z = W @ x + b` and `a = activation(z)`. Repeating this operation across layers produces a prediction. The first notebook makes these steps explicit so you can inspect each intermediate value and shape.

### Backpropagation and learning

Backpropagation applies the chain rule to compute how a loss changes with each weight and bias. An optimizer uses those gradients to update the parameters; a gradient-descent step has the form `parameter = parameter - learning_rate * gradient`. The XOR notebook implements forward passes and parameter updates directly in NumPy. Pay attention to its error sign convention when comparing the update expressions with the gradient-descent formula.

### Activation functions and vanishing gradients

Nonlinear activations let a network represent more than a single linear transformation. Sigmoid, tanh, and ReLU behave differently, especially through their derivatives. When many small derivatives multiply during backpropagation, gradients can become very small, making early layers learn slowly. The activation notebook uses plots to explore these behaviors.

### Regression and classification

Regression predicts numerical values; classification predicts categories. These tasks need suitable output layers and loss functions. The Keras lessons move from predicting concrete strength to classifying MNIST digits. Follow how inputs are prepared, labels are represented, and loss changes during training.

### Convolutional neural networks

Convolutional layers apply learned filters across an image, preserving spatial relationships that flattening removes. Pooling reduces spatial dimensions, and dense layers turn extracted features into class predictions. The CNN lesson applies this pattern to MNIST.

### Attention and transfer learning

Attention builds a context from relevant sequence positions, allowing a decoder to use information across encoder outputs. Transfer learning reuses representations learned on another dataset. The final lessons explore these ideas through sequence modeling, a pretrained VGG16 image classifier base, and BLIP caption generation using Hugging Face Transformers and PyTorch.

### The Keras workflow

Building a model defines its layers and connections. `compile` configures its optimizer, loss, and metrics; `fit` trains it; `evaluate` measures its performance; and `predict` produces outputs for new inputs. The notebooks use both `keras` and `tensorflow.keras` imports, preserving the styles of their examples. The [Keras Sequential guide](https://keras.io/guides/sequential_model/) explains the layer-stack API used in the regression and classification lessons.

## Data and Runtime Notes

- **NumPy lessons:** forward propagation, backpropagation, and activation functions use data or values defined inside the notebooks.
- **Regression:** downloads a concrete-strength CSV from IBM-hosted course storage.
- **MNIST lessons:** load the dataset through Keras, which may download it on first use.
- **Sequence modeling:** defines five English–Spanish sentence pairs directly in the notebook. This is a small teaching example, not an evaluated translation system.
- **Classification and captioning:** downloads an aircraft-damage dataset and pretrained model assets, including VGG16 weights and `Salesforce/blip-image-captioning-base`. It requires network access and more runtime resources than the NumPy lessons.

Training results can vary with random initialization and the runtime environment. The backpropagation example currently uses 180,000 training iterations, so allow it time to finish. Existing notebook outputs are illustrative and do not certify compatibility with every future Colab runtime.

These are teaching examples, not production benchmarks. For experiments that compare architectures or hyperparameters, keep a separate test set that is not used to select the model.

## Contributing

Corrections, clearer explanations, reproducibility reports, and focused notebook improvements are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and validation expectations, and follow the [Code of Conduct](CODE_OF_CONDUCT.md) in project spaces. Coding agents should also read [AGENTS.md](AGENTS.md).

Use [issues](https://github.com/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras/issues) to report reproducible problems or discuss proposed improvements. Releases are generated from Conventional Commit messages through the repository's release workflow; that workflow does not execute the notebooks.

## Conclusion

Work through the examples by inspecting shapes, tracing calculations, and changing one assumption at a time. The early NumPy lessons provide the foundation for understanding what the later Keras training calls automate. Use the exercises to test your understanding and explain why a change affects the result.

## References

- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html): hosted notebook execution, saving work, and runtime limits.
- [Keras Sequential guide](https://keras.io/guides/sequential_model/): building and using stacks of layers.
- The individual notebooks include lesson-specific explanations and links to data or model assets. Preserve existing source acknowledgments when contributing.

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for the full text. External datasets and pretrained models remain subject to their respective terms.
