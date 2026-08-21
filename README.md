# Deep Learning and Neural Networks with Keras

A hands-on curriculum of eight self-contained notebooks: build a neural network by hand in NumPy, then rebuild it in Keras, then scale it up to CNNs, attention, and transfer learning.

![License](https://img.shields.io/github/license/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras)
![Latest Release](https://img.shields.io/github/v/release/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras)
![Conventional Commits](https://img.shields.io/badge/commits-Conventional%20Commits-fa6673)

## What This Repo Is

Eight Jupyter notebooks under `notebooks/`, meant to be worked through in order, that build up deep learning intuition from first principles to a full applied project. The arc:

1. **Hand-rolled NumPy networks** — implement forward propagation, then backward propagation (gradient descent, from scratch, no framework) on tiny problems (XOR) where you can see every number.
2. **Activation and gradient theory** — why sigmoid saturates, why ReLU is cheap but can "die," and what vanishing gradients actually look like in the derivative.
3. **Keras `Sequential` basics** — a regression model (concrete strength) and a classification model (MNIST digits) using plain `Dense` layers.
4. **Convolutional neural networks** — the same MNIST problem, but preserving 2D spatial structure with `Conv2D` / `MaxPooling2D` instead of flattening it away.
5. **Attention and sequence-to-sequence models** — an LSTM encoder-decoder with a hand-written attention layer, translating a tiny English-to-Spanish phrase list.
6. **A capstone project** — transfer learning with a frozen, pretrained VGG16 for image classification, plus a BLIP-based image captioning model bridging PyTorch and TensorFlow.

There is **no application code, no package, and no test suite** in this repository. Each notebook is a standalone lesson: open it, run its cells top to bottom, and it works on its own — nothing here imports from anything else in the repo.

## How to Use This Repo

Work through the notebooks **in the order given** in the [Curriculum & Progress](#curriculum--progress) table — later notebooks assume the concepts (not the code) from earlier ones, even though every notebook still *runs* standalone. If you already know a topic cold, skim its deep dive and its Self-Check questions below before skipping ahead.

For each notebook:

1. Read its [deep dive](#notebook-deep-dives) section here first — it explains the mechanism, not just the API.
2. Open the notebook (linked from both the progress table and the deep dive) and run it top to bottom.
3. Attempt the **Practice Exercises in This Notebook** listed in the deep dive — some are genuinely blank cells, some have solutions hidden in collapsed/commented cells.
4. Answer the three **Self-Check** questions from memory, without looking at the notebook. If you can't, re-read "The Core Idea" and "The Math" before moving on.

Track your own progress by updating the `Status` column in the [Curriculum & Progress](#curriculum--progress) table and the matching `**Status:**` line in each deep dive — flip `⬜ Not Started` to `✅ Completed` as you finish each notebook.

## Environment Setup

All notebooks except `backward-propagation.ipynb` (see the Pyodide note below) expect a standard Python 3 environment with Jupyter and the packages below. The individual notebooks pin slightly different, overlapping versions of `numpy`, `pandas`, `tensorflow_cpu`, etc. (see the notebook-by-notebook pins in each deep dive); the install line below reconciles all of those onto one consistent, newest-compatible set that satisfies every notebook.

**1. Create and activate a virtual environment.**

Use a **Python 3.9–3.12** interpreter (3.11 or 3.12 recommended) — this ceiling comes from the `tensorflow-cpu==2.18.0` and `torch==2.2.0+cpu` wheels pinned below only publishing for cp39–cp312, not from anything in the notebook code (`forward-propagation.ipynb` itself, needing only `numpy`, runs fine on newer interpreters). If your default `python3` resolves to 3.13+, point `venv` at a specific interpreter instead:

Using `venv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

Or using `conda`:

```bash
conda create -n dl-keras python=3.11
conda activate dl-keras
```

**2. Install the core dependencies** (covers notebooks 1, 3, 4, 5, 6, 7 and the TensorFlow half of notebook 8):

```bash
pip install numpy==2.0.2 pandas==2.2.3 matplotlib==3.9.2 tensorflow-cpu==2.18.0 pillow==11.1.0 transformers==4.38.2 jupyterlab
```

**3. Install CPU PyTorch separately** (only needed for `classification-and-captioning.ipynb`, notebook 8 — its BLIP captioning half runs on PyTorch/Transformers, not Keras). PyTorch's CPU wheels live on their own index, so this needs its own command:

```bash
pip install torch==2.2.0+cpu torchvision==0.17.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

(Notebook 8's own install cell also pins `torchaudio==2.2.0+cpu` alongside these; it's omitted here because nothing in the notebook imports it, and skipping it keeps the download smaller.)

**4. Launch Jupyter:**

```bash
jupyter lab
```

### A note on `backward-propagation.ipynb`'s kernel

`backward-propagation.ipynb` is authored for a **`Python (Pyodide)`** kernel — the in-browser, JupyterLite-style runtime that ships with numpy and matplotlib preinstalled and needs no local `pip install` at all. If you open it in JupyterLite (or any Pyodide-backed notebook environment), it runs as-is. It also runs fine on a normal local `ipykernel` as long as `numpy` and `matplotlib` are installed — the pinned versions from step 2 above are sufficient; there is nothing Pyodide-specific in the code itself.

### A note on the notebooks' own `!pip install` cells

Several notebooks include their own `!pip install ...` cells at the top. In most of them (`forward-propagation.ipynb`, `regression-with-keras.ipynb`, `classification-with-keras.ipynb`, `convolutional-neural-networks-with-keras.ipynb`, `transformers-with-Keras.ipynb`) these are **commented out** — they're left as a reference for "this is what was installed," not something that runs. The one exception is **`activation-functions-and-vanishing.ipynb`**, where the `!pip install numpy==2.0.2` and `!pip install matplotlib==3.9.2` cells are **live** and will actually reinstall those packages every time you run the notebook top to bottom. If you're on a locked-down or offline environment, comment those two cells out after your first run — they duplicate what step 2 above already installed.

## Curriculum & Progress

| # | Notebook | Topic | Status | Core Libraries |
|---|---|---|---|---|
| 1 | [`forward-propagation.ipynb`](notebooks/forward-propagation.ipynb) ([deep dive](#1-forward-propagationipynb--forward-propagation)) | Forward propagation from scratch | ✅ Completed | numpy |
| 2 | [`backward-propagation.ipynb`](notebooks/backward-propagation.ipynb) ([deep dive](#2-backward-propagationipynb--backward-propagation)) | Backward propagation from scratch (XOR) | ⬜ Not Started | numpy, matplotlib |
| 3 | [`activation-functions-and-vanishing.ipynb`](notebooks/activation-functions-and-vanishing.ipynb) ([deep dive](#3-activation-functions-and-vanishingipynb--activation-functions--vanishing-gradients)) | Activation functions & vanishing gradients | ⬜ Not Started | numpy, matplotlib |
| 4 | [`regression-with-keras.ipynb`](notebooks/regression-with-keras.ipynb) ([deep dive](#4-regression-with-kerasipynb--regression-models-with-keras)) | Regression with Keras `Sequential` | ⬜ Not Started | keras, pandas |
| 5 | [`classification-with-keras.ipynb`](notebooks/classification-with-keras.ipynb) ([deep dive](#5-classification-with-kerasipynb--classification-models-with-keras)) | Classification on MNIST | ⬜ Not Started | keras |
| 6 | [`convolutional-neural-networks-with-keras.ipynb`](notebooks/convolutional-neural-networks-with-keras.ipynb) ([deep dive](#6-convolutional-neural-networks-with-kerasipynb--convolutional-neural-networks-with-keras)) | CNNs on MNIST | ⬜ Not Started | keras |
| 7 | [`transformers-with-Keras.ipynb`](notebooks/transformers-with-Keras.ipynb) ([deep dive](#7-transformers-with-kerasipynb--seq2seq-with-lstm--attention)) | Seq2seq with LSTM + attention | ⬜ Not Started | keras, tensorflow.keras |
| 8 | [`classification-and-captioning.ipynb`](notebooks/classification-and-captioning.ipynb) ([deep dive](#8-classification-and-captioningipynb--capstone-aircraft-damage-classification--captioning)) | Capstone: transfer learning + captioning | ⬜ Not Started | keras, tensorflow, torch, transformers |

## Notebook Deep Dives

### 1. `forward-propagation.ipynb` — Forward Propagation

**Status:** ✅ Completed
**Kernel:** Python 3 (ipykernel)
**Prerequisites:** none · concept: [weighted sum](#concepts-glossary)
**Open:** [`notebooks/forward-propagation.ipynb`](notebooks/forward-propagation.ipynb)

**What You'll Learn**
- How a single artificial neuron turns inputs into an output: a weighted sum plus a bias, passed through an activation function.
- How to chain neurons into layers, and layers into a full network, by hand.
- How to generalize the by-hand computation into reusable functions that build and run a network of arbitrary shape.
- Why forward propagation alone — with no training — is a useful thing to isolate and understand first.

**The Core Idea**
Every neuron does the same two-step job: multiply each input by a learned weight, sum those products and add a bias (the **weighted sum**), then squash that number through a nonlinear **activation function** (here, sigmoid) to produce the neuron's output. Forward propagation is just doing this, layer by layer, feeding each layer's outputs in as the next layer's inputs, until you reach the final output layer. This notebook first does it with explicit variables (`z_11`, `a_11`, `z_12`, `a_12`, `z_2`, `a_2`) for a tiny 2-input/2-hidden-node/1-output network so every number is visible, then generalizes the same steps into functions that work for a network of any size. There is deliberately no loss function and no gradient anywhere in this notebook — it isolates *inference* from *learning*, which is covered starting in notebook 2.

**The Math**
For a single neuron with inputs $x_i$, weights $w_i$, and bias $b$, the weighted sum is:

$$z = \sum_i w_i x_i + b$$

That weighted sum is then passed through the sigmoid activation:

$$a = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Sigmoid squashes any real-valued $z$ into the open interval $(0, 1)$, which is why it was historically popular for representing "activation strength" or probability-like outputs.

**Key API / Functions**

| Name | Signature | Does |
|---|---|---|
| `initialize_network` | `initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output)` | Builds a nested dict `network[layer_name][node_name] = {'weights': array, 'bias': array}` with random initial values |
| `compute_weighted_sum` | `compute_weighted_sum(inputs, weights, bias)` | Returns `np.sum(inputs * weights) + bias` |
| `node_activation` | `node_activation(weighted_sum)` | Returns `1.0 / (1.0 + np.exp(-1 * weighted_sum))` (sigmoid) |
| `forward_propagate` | `forward_propagate(network, inputs)` | Loops layer by layer, prints each layer's outputs, returns final `network_predictions` |

**Common Pitfalls & Known Issues**
- Weights and biases are randomly initialized on every run, so your printed numbers will differ from anyone else's — that's expected, not a bug.
- The two embedded lesson images are hosted via plain-HTTP `cocl.us` short links; if they fail to load, the surrounding markdown text still fully explains the diagram.
- `compute_weighted_sum` expects `inputs` and `weights` as same-length 1D arrays — mismatched lengths will broadcast-error rather than silently truncate.

**Practice Exercises in This Notebook**
None — this notebook is a guided walkthrough with every cell already filled in and its outputs saved. There are no blank cells to complete.

**Self-Check**
1. What two operations does a single neuron perform between receiving its inputs and producing its output?
2. If a neuron has inputs `[1.0, 2.0]`, weights `[0.5, -0.5]`, and bias `0.1`, what is its weighted sum $z$ before activation?
3. Why does this notebook never compute a loss or a gradient — what specifically is it teaching in isolation?

### 2. `backward-propagation.ipynb` — Backward Propagation

**Status:** ⬜ Not Started
**Kernel:** Python (Pyodide)
**Prerequisites:** [§1 Forward Propagation](#1-forward-propagationipynb--forward-propagation) · concept: [backpropagation](#concepts-glossary)
**Open:** [`notebooks/backward-propagation.ipynb`](notebooks/backward-propagation.ipynb)

**What You'll Learn**
- Why a single-layer network cannot solve XOR, and how a hidden layer fixes that.
- How error at the output layer is pushed backward through the network to produce a gradient for every weight.
- How the chain rule turns "the output was wrong by this much" into "this specific weight should change by this much."
- How the choice of learning rate and epoch count trades off training speed against convergence.

**The Core Idea**
This notebook trains a 2-input/2-hidden/1-output network on the **XOR gate** — the classic example a single layer of weights *cannot* represent, because XOR's positive and negative examples aren't separable by any straight line. A hidden layer with a nonlinearity fixes that. Training works by **backpropagation**: run the network forward to get a prediction, measure how wrong it was, then walk *backward* through the network computing how much each weight contributed to that error, using the chain rule at every layer. Each weight is nudged in the direction that reduces the error, scaled by the **learning rate**. Note the data shape: `X` is `2×4` with **each column, not row, one training example** — that column-major convention is why every `np.dot` in this notebook is arranged the way it is (weight matrices multiply from the left), and it's worth tracing through once by hand so the shapes stop feeling arbitrary.

**The Math**
Forward pass through both layers uses sigmoid activations $a_1 = \sigma(z_1)$, $a_2 = \sigma(z_2)$, same as notebook 1. Training adds the backward pass. First, the output error and its local gradient:

$$\text{error} = d - a_2$$

$$da_2 = \text{error} \odot a_2 \odot (1 - a_2)$$

The second factor is the sigmoid derivative $\sigma'(z) = \sigma(z)(1-\sigma(z))$, applied elementwise ($\odot$) — this is the chain rule connecting the loss to the pre-activation $z_2$.

That gradient is pushed back through the second layer's weights (transposed, since we're now going backward) to find the hidden layer's share of the error:

$$da_1 = w_2^T \cdot dz_2$$

$$dz_1 = da_1 \odot a_1 \odot (1 - a_1)$$

Finally, every weight and bias is updated in the direction that reduces error, scaled by the learning rate $lr$:

$$w_2 \mathrel{+}= lr \cdot (dz_2 \cdot a_1^T), \qquad b_2 \mathrel{+}= lr \cdot \sum dz_2$$

**Key API / Functions**

| Name | Signature | Does |
|---|---|---|
| `initialize_network_parameters` | `initialize_network_parameters()` | Returns `w1, b1, w2, b2, lr, epochs` with weights random in $[-1, 1]$, `lr=0.1`, `epochs=180000` |
| `np.dot(w2.T, dz2)` | — | Pushes the output layer's error gradient backward through the (transposed) second-layer weights to the hidden layer |
| `np.random.rand(...) * 2 - 1` | — | Rescales `[0, 1)` uniform randoms into `[-1, 1)` for weight initialization |

**Common Pitfalls & Known Issues**
- `X`'s shape is `2×4` with examples as **columns**; if you reshape or index it as if examples were rows, every subsequent `np.dot` will silently compute the wrong thing (shapes will often still "work," just mean something else).
- `epochs = 180000` is large; error is only recorded every 10,000 epochs into `error_list`, so the loss plot at the end is coarse-grained by design, not a bug.
- Heading levels in the source markdown are inconsistent (mixing `#` and `##`) — a rendering quirk of the original notebook, not a code issue.

**Practice Exercises in This Notebook**
Two genuinely blank cells (`# Write your code here`): (1) implement the same backpropagation training loop for an **AND** gate instead of XOR; (2) retrain the XOR network with `lr=0.01` and `epochs=1000000`, and compare convergence behavior to the original `lr=0.1`/`epochs=180000` run.

**Self-Check**
1. Why can't a network with no hidden layer learn XOR, but this 2-hidden-node network can?
2. In `da2 = error * (a2 * (1 - a2))`, what is the second factor, and why does the chain rule require multiplying by it?
3. If you transposed `X` to be `4×2` (examples as rows) without changing any of the `np.dot` calls, would the notebook still train correctly? Why or why not?

### 3. `activation-functions-and-vanishing.ipynb` — Activation Functions & Vanishing Gradients

**Status:** ⬜ Not Started
**Kernel:** Python 3 (ipykernel)
**Prerequisites:** [§2 Backward Propagation](#2-backward-propagationipynb--backward-propagation) · concept: [vanishing gradient](#concepts-glossary)
**Open:** [`notebooks/activation-functions-and-vanishing.ipynb`](notebooks/activation-functions-and-vanishing.ipynb)

**What You'll Learn**
- Why backpropagation's chain rule makes the *derivative* of an activation function just as important as the function itself.
- Why sigmoid saturates for large-magnitude inputs, and why that causes vanishing gradients in deep networks.
- Why ReLU largely avoids vanishing gradients, and what a "dead neuron" is instead.
- How tanh compares to sigmoid and ReLU on both range and saturation behavior.

**The Core Idea**
Backpropagation (notebook 2) multiplies gradients together across layers via the chain rule — and each of those factors is an activation function's *derivative*. Sigmoid's derivative is largest (0.25) at $z=0$ and shrinks toward 0 as $|z|$ grows in either direction, because sigmoid **saturates** (flattens out) far from zero. Stack enough layers of sigmoid and those small derivatives multiply together into a vanishingly small gradient at the earliest layers — those layers barely learn at all. This is the **vanishing gradient problem**. ReLU sidesteps it: its derivative is a constant 1 for any positive input, so gradients pass through unshrunk no matter how deep the stack. The tradeoff is the **dead neuron** problem — a ReLU unit that ends up with a negative weighted sum for every training example outputs 0 and has 0 gradient, so it can never update again. tanh sits in between: like sigmoid it saturates at its extremes, but it's zero-centered with range $(-1, 1)$ instead of $(0, 1)$, which in practice gives it slightly better-conditioned gradients than sigmoid.

**The Math**
Sigmoid and its derivative:

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)\,(1 - \sigma(z))$$

ReLU and its derivative:

$$\text{ReLU}(z) = \max(0, z), \qquad \text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}$$

tanh and its derivative:

$$\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}, \qquad \tanh'(z) = 1 - \tanh(z)^2$$

**Key API / Functions**

| Name | Signature | Does |
|---|---|---|
| `sigmoid` | `sigmoid(z)` | `1 / (1 + np.exp(-z))` |
| `sigmoid_derivative` | `sigmoid_derivative(z)` | `sigmoid(z) * (1 - sigmoid(z))` |
| `relu` | `relu(z)` | `np.maximum(0, z)` |
| `relu_derivative` | `relu_derivative(z)` | `np.where(z > 0, 1, 0)` |
| `tanh` / `tanh_derivative` | `tanh(z)` / `tanh_derivative(z)` | `np.tanh(z)` / `1 - np.tanh(z)**2` |

**Common Pitfalls & Known Issues**
- The `!pip install numpy==2.0.2` and `!pip install matplotlib==3.9.2` cells in this notebook are **live**, not commented out — every run reinstalls those packages (see [Environment Setup](#environment-setup)).
- Both "Practice Exercises" are already fully filled in, including their "double-click for the solution" reveal cells, which just duplicate the code already shown above them — they aren't real blank exercises, so don't expect anything to be missing.
- The source markdown contains a typo, "vausing vanishing gradient problem" — harmless, just don't search for that exact phrase expecting a different section.

**Practice Exercises in This Notebook**
Both exercises are pre-filled (not blank): the first compares sigmoid vs. ReLU over `z = np.linspace(-10, 10, 400)`, the second compares tanh vs. ReLU over `np.linspace(-5, 5, 100)`. Read through the pre-filled code rather than expecting cells to complete.

**Self-Check**
1. Why does stacking many sigmoid layers make the vanishing gradient problem *worse* as depth increases, specifically in terms of the chain rule?
2. At $z = 0$, what is $\sigma'(z)$, and how does that compare to $\text{ReLU}'(z)$ at $z=0.001$ versus $z=-0.001$?
3. A colleague says "ReLU has no gradient problems at all." What specific failure mode are they missing?

### 4. `regression-with-keras.ipynb` — Regression Models with Keras

**Status:** ⬜ Not Started
**Kernel:** Python 3 (ipykernel)
**Prerequisites:** [§3 Activation Functions & Vanishing Gradients](#3-activation-functions-and-vanishingipynb--activation-functions--vanishing-gradients) · concept: [loss function](#concepts-glossary)
**Open:** [`notebooks/regression-with-keras.ipynb`](notebooks/regression-with-keras.ipynb)

**What You'll Learn**
- How to go from raw tabular data to a trained Keras model without hand-writing forward or backward propagation.
- Why regression outputs need a linear (no-activation) final layer, unlike the sigmoid/softmax outputs used for classification.
- Why normalizing input features before training matters for gradient descent.
- How to read training/validation loss curves to judge whether a model is learning.

**The Core Idea**
Everything you implemented by hand in notebooks 1–2 — weighted sums, activations, gradients, weight updates — is what `keras.Sequential` does internally when you call `.compile()` and `.fit()`. This notebook applies that machinery to a real regression task: predicting concrete compressive strength from 8 numeric features (cement content, water, age, etc.). Because the target is a continuous number rather than a class, the output layer is a single `Dense(1)` unit with **no activation function** — you want the raw weighted sum as the prediction, not something squashed into $(0,1)$. The features are also **z-score normalized** (each column has its mean subtracted and is divided by its standard deviation) before training, because gradient descent converges much more reliably when input features are on comparable scales — a feature ranging in the thousands would otherwise dominate the loss landscape over one ranging in single digits.

**The Math**
The loss being minimized is mean squared error over $n$ examples, comparing predictions $\hat y_i$ to true strengths $y_i$:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat y_i)^2$$

Z-score normalization of a feature column $x$ with mean $\mu$ and standard deviation $\sigma$:

$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

**Key API / Functions**

| Name | Signature | Does |
|---|---|---|
| `Sequential` | `keras.Sequential([...])` | Stacks layers linearly into a model |
| `Input` | `Input(shape=(n_cols,))` | Declares the model's input shape explicitly |
| `Dense` | `Dense(units, activation=...)` | Fully connected layer; hidden layers use `'relu'`, the output layer uses no activation (linear) |
| `model.compile` | `model.compile(optimizer='adam', loss='mean_squared_error')` | Configures the optimizer and loss before training |
| `model.fit` | `model.fit(X, y, validation_split=..., epochs=..., verbose=2)` | Trains the model, holding out a validation split each epoch |

**Common Pitfalls & Known Issues**
- The dataset is pulled live from a hosted CSV URL at notebook run time — no internet access means this notebook can't run past the data-loading cell.
- Forgetting to normalize (or normalizing only the training set with training statistics but not applying the *same* mean/std to any held-out data) is a common source of silently poor convergence.
- The `TF_ENABLE_ONEDNN_OPTS` / `TF_CPP_MIN_LOG_LEVEL` environment variables at the top only silence TensorFlow's console logging — they have no effect on training behavior or results.

**Practice Exercises in This Notebook**
Solutions are hidden in HTML comments in the source (not visibly blank in the rendered notebook, but intended to be attempted first): (1) build a deeper model with 5 hidden layers instead of the original's fewer layers; (2) retrain with `validation_split=0.1` and `epochs=100`, and compare the loss curve to the original run.

**Self-Check**
1. Why must the output layer for this regression task have no activation function, in contrast to a classification output layer?
2. What would likely happen to training if the input features were left unnormalized — cement content in the hundreds, superplasticizer in single digits?
3. Is a lower training loss than validation loss always a good sign? What does a *growing* gap between the two usually indicate?

### 5. `classification-with-keras.ipynb` — Classification Models with Keras

**Status:** ⬜ Not Started
**Kernel:** Python 3 (ipykernel)
**Prerequisites:** [§4 Regression with Keras](#4-regression-with-kerasipynb--regression-models-with-keras) · concept: [softmax](#concepts-glossary)
**Open:** [`notebooks/classification-with-keras.ipynb`](notebooks/classification-with-keras.ipynb)

**What You'll Learn**
- How to prepare image data (MNIST digits) for a dense network by flattening and normalizing pixel values.
- How one-hot encoding turns integer class labels into vectors a softmax output can be trained against.
- Why classification outputs use softmax instead of the linear output from notebook 4's regression model.
- How to save and reload a trained Keras model in the modern `.keras` format.

**The Core Idea**
This notebook is the classification counterpart to notebook 4's regression model, applied to the MNIST handwritten-digit dataset (60,000 training images, 10,000 test images, each 28×28 grayscale pixels). Every image is **flattened** from a 28×28 grid into a single 784-length vector, since a plain `Dense` network has no notion of 2D spatial structure — it just sees 784 independent numbers (this flattening, and what's lost by doing it, is exactly what notebook 6's convolutional layers are built to avoid). Pixel values are divided by 255 to land in $[0,1]$, and integer labels (0–9) are **one-hot encoded** into 10-length vectors so they can be compared against the model's 10-way probability output. The final layer is `Dense(10, activation='softmax')`, which turns raw output scores into a probability distribution over the 10 digit classes that sums to 1.

**The Math**
Softmax converts a vector of raw scores $z$ into probabilities for class $i$ out of $K$ classes:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{k=1}^{K} e^{z_k}}$$

The training loss, categorical cross-entropy, penalizes the model for putting low probability mass on the true class $y$ (one-hot, so only the true class's term is nonzero):

$$\text{CE} = -\sum_{i=1}^{K} y_i \log(\hat y_i)$$

**Key API / Functions**

| Name | Signature | Does |
|---|---|---|
| `mnist.load_data` | `mnist.load_data()` | Returns `(X_train, y_train), (X_test, y_test)` |
| `reshape` | `X.reshape(n, 784).astype('float32')` | Flattens each 28×28 image into a 784-length vector |
| `to_categorical` | `to_categorical(y)` | One-hot encodes integer labels into class-probability-shaped vectors |
| `model.save` / `keras.saving.load_model` | `model.save('name.keras')` / `load_model('name.keras')` | Persists/restores a full model in the modern `.keras` format |

**Common Pitfalls & Known Issues**
- Training for the full `epochs=10` can take **over 20 minutes on CPU** by the notebook's own admission — plan accordingly, or reduce epochs for a quick smoke test.
- Flattening discards all spatial adjacency information (a pixel no longer "knows" its neighbors) — this is a deliberate simplification for this notebook, corrected in notebook 6.
- **Defect:** the Practice Exercise 2 solution cell has a stray trailing `|` after its final `print(...)` call, causing a `SyntaxError` if run as-is — see [Known Notebook Defects](#known-notebook-defects).

**Practice Exercises in This Notebook**
(1) Build a deeper 6-`Dense`-layer network and compare its accuracy to the original 3-layer model. (2) Reload the original 3-layer model that was saved earlier in the notebook (`keras.saving.load_model('classification_model.keras')`) and continue training it for 10 more epochs as a warm start, rather than retraining from scratch — note this reloads the original model, not the 6-layer model built in Exercise 1, which is never saved.

**Self-Check**
1. Why does softmax's output always sum to exactly 1 across the 10 classes, and why is that property necessary for the cross-entropy loss?
2. What specific information about each digit image is discarded by `reshape(n, 784)`, and why doesn't that matter for a purely `Dense` network?
3. If you skipped `to_categorical` and trained directly on integer labels 0–9 with `categorical_crossentropy`, what would go wrong?

### 6. `convolutional-neural-networks-with-keras.ipynb` — Convolutional Neural Networks with Keras

**Status:** ⬜ Not Started
**Kernel:** Python 3 (ipykernel)
**Prerequisites:** [§5 Classification with Keras](#5-classification-with-kerasipynb--classification-models-with-keras) · concept: [pooling](#concepts-glossary)
**Open:** [`notebooks/convolutional-neural-networks-with-keras.ipynb`](notebooks/convolutional-neural-networks-with-keras.ipynb)

**What You'll Learn**
- Why keeping images in their native 2D shape (instead of flattening, as notebook 5 did) lets a network exploit spatial structure.
- How a convolutional filter slides across an image to produce a feature map.
- What max pooling does and why it's typically paired with convolution.
- How stacking multiple conv/pool blocks builds up from low-level to higher-level visual features.

**The Core Idea**
Notebook 5 flattened each MNIST digit into a 784-length vector, throwing away the fact that pixel $(i, j)$ is spatially next to pixel $(i, j{+}1)$. This notebook keeps images in their native `(28, 28, 1)` shape and uses `Conv2D` layers instead: each filter is a small learned weight grid (e.g. 5×5) that slides across the image computing a weighted sum at every position, producing a **feature map** that lights up wherever that local pattern (an edge, a curve) appears. `MaxPooling2D` then shrinks each feature map by keeping only the maximum value in each small window, which reduces the data's size while keeping the strongest signal — and makes the network somewhat tolerant to small shifts in exactly where a feature appears. Stacking a second `Conv2D` + `MaxPooling2D` block on top of the first lets later filters combine the first block's simple features (edges) into more complex ones (corners, loops) before a final `Flatten` + `Dense` head classifies the digit — this is the first notebook to flatten only at the very end, after the spatial processing is done. The notebook builds this twice: **Model A** has a single `Conv2D` + `MaxPooling2D` block, and **Model B** adds a second, smaller `Conv2D` + `MaxPooling2D` block on top of it — the two are trained and compared side by side.

**The Math**
For a square input of size $n \times n$, a filter (kernel) of size $f \times f$, stride $s$, and padding $p$, the output feature map's spatial size is:

$$\text{output size} = \left\lfloor \frac{n - f + 2p}{s} \right\rfloor + 1$$

With no padding ($p=0$), a 28×28 input, a 5×5 filter, and stride 1, the output is $\lfloor (28 - 5)/1 \rfloor + 1 = 24$, giving a 24×24 feature map.

**Key API / Functions**

| Name | Signature | Does |
|---|---|---|
| `Conv2D` | `Conv2D(filters, kernel_size, strides=(1,1), activation='relu')` | Learns `filters` sliding-window feature detectors over the input |
| `MaxPooling2D` | `MaxPooling2D(pool_size=(2,2), strides=(2,2))` | Downsamples by taking the max value in each pooling window |
| `Flatten` | `Flatten()` | Collapses the final spatial feature maps into a 1D vector for the `Dense` head |

**Common Pitfalls & Known Issues**
- Input must be reshaped to 4D `(samples, 28, 28, 1)` — the trailing channel dimension is easy to forget and will raise a shape error in the first `Conv2D` layer.
- This is **the first notebook to introduce `batch_size`** explicitly (200) — omitting it falls back to Keras's default (32), which changes both training speed and the exact gradient noise per step.
- The notebook `!pip install`s `pandas` and `matplotlib` that it never actually imports or uses — harmless, just unnecessary.

**Practice Exercises in This Notebook**
Genuinely blank cells (`# Write your answer here`), solutions hidden in HTML comments: (1) retrain Model A or B with `batch_size=1024` and compare speed/accuracy to `batch_size=200`; (2) retrain with `batch_size=1024` and `epochs=25`.

**Self-Check**
1. What specific structural information does `Conv2D` preserve that flattening (notebook 5) destroys?
2. Using the output-size formula, what is the output size of a 28×28 input with a 2×2 filter, stride 1, no padding?
3. A teammate says "pooling just makes training faster by shrinking the data." What else does max pooling provide beyond speed?

### 7. `transformers-with-Keras.ipynb` — Seq2Seq with LSTM + Attention

**Status:** ⬜ Not Started
**Kernel:** Python 3 (ipykernel)
**Prerequisites:** [§6 CNNs with Keras](#6-convolutional-neural-networks-with-kerasipynb--convolutional-neural-networks-with-keras) · concept: [attention (self- vs cross-)](#concepts-glossary)
**Open:** [`notebooks/transformers-with-Keras.ipynb`](notebooks/transformers-with-Keras.ipynb)

**What You'll Learn**
- How an encoder-decoder (seq2seq) architecture translates a sequence into another sequence of a different length.
- What teacher forcing is and why it speeds up training a decoder.
- How scaled dot-product attention lets a decoder look back at *every* encoder position instead of relying on a single fixed-size summary vector.
- How to implement a custom trainable Keras `Layer` from scratch, including its own weights.

**The Core Idea**
This notebook builds a tiny English-to-Spanish translator on a hardcoded 5-pair phrase list. An **encoder** LSTM reads the source sentence and produces a final hidden state (`state_h`, `state_c`) summarizing it; a **decoder** LSTM is initialized with that state and, at each step, predicts the next output word. During training the decoder uses **teacher forcing**: instead of feeding its own (possibly wrong) previous prediction back in, it's fed the *actual* previous word from the target sentence at every step, which keeps training stable and fast since one bad early prediction can't cascade into garbage for the rest of the sequence. On top of the base encoder-decoder, a hand-written `SelfAttention` layer lets the decoder, at every output step, look back across *all* of the encoder's hidden states and weight them by relevance, rather than being forced through the single fixed-size summary vector — this is what lets the model handle longer, more nuanced sentences than a plain encoder-decoder could.

A teaching note worth flagging explicitly: despite its name, `SelfAttention` in this notebook actually implements **cross-attention**, not self-attention — its queries come from the *decoder's* current state while its keys and values come from the *encoder's* outputs, i.e. one sequence attending to a different sequence, not a sequence attending to itself (see "attention (self- vs cross-)" in the [Concepts Glossary](#concepts-glossary)). The notebook's markdown also documents a `compute_output_shape` method that the code never actually implements — the layer works fine without it in this setup (Keras can infer the output shape at runtime), but if you're using this class as a template elsewhere, don't assume that method exists just because the prose describes it.

This notebook's imports are **deliberately mixed** between `keras.*` and `tensorflow.keras.*` in the same cells — for example `from keras.models import Model` sits alongside `from tensorflow.keras import backend as K`, and a later cell re-imports `Layer`/`Model` from `tensorflow.keras`, shadowing the bare-`keras` versions imported earlier. This is intentional to preserve exactly as written, not something to "clean up" — see the repo's `CLAUDE.md`.

**The Math**
Scaled dot-product attention computes, for query matrix $Q$, key matrix $K$, and value matrix $V$, with key dimension $d_k$:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right) V$$

The $QK^T$ term scores how relevant each key is to each query (higher dot product = more aligned); dividing by $\sqrt{d_k}$ keeps those scores from growing too large as dimensionality increases (which would push softmax into a near-one-hot, hard-to-train regime); softmax turns the scores into weights summing to 1; multiplying by $V$ produces a weighted blend of the value vectors.

**Key API / Functions**

| Name | Signature | Does |
|---|---|---|
| `Tokenizer` | `Tokenizer()` | Builds a word-to-integer vocabulary from the text corpus |
| `pad_sequences` | `pad_sequences(sequences, ...)` | Pads variable-length sequences to a common length |
| `LSTM(..., return_sequences=True, return_state=True)` | — | Returns both the full output sequence and the final `(state_h, state_c)` for use as another LSTM's initial state |
| `SelfAttention.build` / `.call` | custom `Layer` subclass | Creates trainable `Wq`, `Wk`, `Wv` weights in `build()`; computes scaled dot-product attention with `K.batch_dot` in `call()` |

**Common Pitfalls & Known Issues**
- The mixed `keras.*` / `tensorflow.keras.*` imports mean a later cell's `Layer`/`Model` is not the same object as an earlier cell's — usually harmless here since both resolve to compatible implementations, but worth knowing about if you see a confusing `isinstance` mismatch elsewhere.
- Training runs 100 epochs at `batch_size=16` on a 5-example dataset — this is a toy-scale demonstration of the *mechanism*, not a model meant to generalize to unseen sentences.
- The `SelfAttention` class name is misleading (see "The Core Idea" above) — this is a teaching note, not a defect, but it's easy to internalize the wrong mental model if you take the class name at face value.

**Practice Exercises in This Notebook**
Solutions hidden in HTML comments: (1) swap the `glorot_uniform` weight initializer for `he_uniform` and compare training; (2) swap the `adam` optimizer for `adagrad` and compare training.

**Self-Check**
1. What problem does teacher forcing solve during training, and what changes at inference time when the true previous word isn't available?
2. In the scaled dot-product attention formula, what does dividing by $\sqrt{d_k}$ prevent, and why does that matter for softmax specifically?
3. Given what `SelfAttention`'s queries, keys, and values actually come from in this notebook, is it really self-attention? What would need to change for it to be self-attention in the strict sense?

### 8. `classification-and-captioning.ipynb` — Capstone: Aircraft Damage Classification + Captioning

**Status:** ⬜ Not Started
**Kernel:** Python 3 (ipykernel)
**Prerequisites:** [§7 Seq2Seq with LSTM + Attention](#7-transformers-with-kerasipynb--seq2seq-with-lstm--attention) · concept: [transfer learning](#concepts-glossary)
**Open:** [`notebooks/classification-and-captioning.ipynb`](notebooks/classification-and-captioning.ipynb)

**What You'll Learn**
- How to reuse a pretrained image model (VGG16, trained on ImageNet) for a completely different, small, task-specific dataset via transfer learning.
- Why freezing a pretrained model's base layers and training only a new head is both faster and less prone to overfitting than training from scratch.
- How to bridge a PyTorch model (BLIP, for captioning) into a TensorFlow/Keras pipeline using a custom layer and `tf.py_function`.
- How task-conditioned prompting changes a single pretrained model's output (a caption vs. a longer summary) without retraining it.

**The Core Idea**
This is the capstone: a two-part hands-on project structured as 10 numbered Tasks, 9 of which are genuinely blank cells you must fill in yourself. **Part 1** classifies aircraft images as showing a "dent" or a "crack" using **transfer learning**: `VGG16`, pretrained on the million-plus-image ImageNet dataset, is loaded with `weights='imagenet'` and `include_top=False` (discarding its original 1000-class output head), and every one of its convolutional base layers is **frozen** (`layer.trainable = False`) so their ImageNet-learned features are reused unchanged. A new head — `Flatten → Dense(512, relu) → Dropout(0.3) → Dense(512, relu) → Dropout(0.3) → Dense(1, sigmoid)` — is trained from scratch on top of those frozen features to make the binary dent-vs-crack decision; the `Adam` learning rate is set deliberately low (`0.0001`) since only a small head is being fit and large updates could overshoot. **Part 2** wraps a pretrained BLIP model (PyTorch, via Hugging Face `transformers`) inside a custom `tf.keras.layers.Layer`, using `tf.py_function` to call into PyTorch code from inside a TensorFlow graph — this is what lets a Keras pipeline call a PyTorch model without rewriting it. The *same* BLIP model produces either a short caption or a longer summary purely by changing its text prompt (`"This is a picture of"` vs. `"This is a detailed photo showing"`) — no retraining, just different conditioning text.

**The Math**
Part 1's classifier is trained with binary cross-entropy, for predicted probability $\hat y \in (0,1)$ against true binary label $y \in \{0,1\}$ (dent vs. crack):

$$\text{BCE} = -\big[y \log(\hat y) + (1-y)\log(1-\hat y)\big]$$

**Key API / Functions**

| Name | Signature | Does |
|---|---|---|
| `VGG16` | `VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))` | Loads pretrained convolutional base, discarding the original classification head |
| `ImageDataGenerator` / `flow_from_directory` | `.flow_from_directory(dir, class_mode='binary', target_size=(224,224))` | Streams and labels images directly from a folder structure |
| `BlipProcessor.from_pretrained` / `BlipForConditionalGeneration.from_pretrained` | `.from_pretrained("Salesforce/blip-image-captioning-base")` | Loads the pretrained BLIP processor and captioning model |
| `tf.py_function` | `tf.py_function(func, inp, Tout)` | Wraps an arbitrary Python (here, PyTorch) call so it can run inside a TF graph/layer |

**Common Pitfalls & Known Issues**
- Downloads are heavy: the aircraft-damage dataset `.tar`, VGG16's ImageNet weights on first use, and the CPU PyTorch wheels/BLIP weights — see [Troubleshooting](#troubleshooting) for what to do if any of these fail partway.
- **Defect:** a config cell ships as `batch_size =` / `n_epochs =` with no values assigned — `SyntaxError`, and it is *not* one of the 10 numbered Tasks, so it's easy to miss and blame the wrong cell. See [Known Notebook Defects](#known-notebook-defects).
- **Defect:** Task 9's cell contains a leftover Colab path, `image_path = "/content/sample_image.jpg"` — `FileNotFoundError` locally. See [Known Notebook Defects](#known-notebook-defects).
- `import zipfile` is unused (the dataset is a `.tar`, extracted with `tarfile`) — harmless, ignore it.
- Imports are mixed three ways in this notebook (bare `keras`, `tensorflow.keras`, and `tf.keras.*` attribute access) — intentional to preserve as written, not something to normalize (see `CLAUDE.md`).

**Practice Exercises in This Notebook**
Tasks 1–2: build the `valid_generator` / `test_generator`. Task 3: load VGG16. Task 4: compile the model. Task 5: train it. Task 6: plot accuracy curves. Task 7: visualize predictions against true labels. Task 8: write the BLIP helper function (the only task with a hidden hint). Task 9: generate a caption for a test image. Task 10: generate a summary for the same image. All 9 are truly blank `# Write your code here` cells required for grading/submission.

**Self-Check**
1. Why freeze VGG16's base layers instead of training the entire network (base + new head) end to end on the small aircraft dataset?
2. Why is the `Adam` learning rate set so low (`0.0001`) for training the new head, compared to a typical from-scratch learning rate?
3. If BLIP is a single, unretrained pretrained model, how can it produce both a short caption and a longer, different summary for the same image?

## Concepts Glossary

- **Activation function** — the nonlinear function (sigmoid, ReLU, tanh, softmax, …) applied to a neuron's weighted sum; without it, stacking layers would collapse to a single linear transformation. See [§1](#1-forward-propagationipynb--forward-propagation), [§3](#3-activation-functions-and-vanishingipynb--activation-functions--vanishing-gradients).
- **Attention (self- vs cross-)** — a mechanism that lets a model weight and blend different positions of a sequence when producing an output; *self*-attention attends within one sequence, *cross*-attention attends from one sequence (e.g. a decoder) to a different one (e.g. an encoder). See [§7](#7-transformers-with-kerasipynb--seq2seq-with-lstm--attention).
- **Backpropagation** — the algorithm that computes how much each weight contributed to the output error by applying the chain rule backward through the network, layer by layer. See [§2](#2-backward-propagationipynb--backward-propagation).
- **Batch size vs. epoch** — batch size is how many training examples are processed before one weight update; an epoch is one full pass over the entire training set (many batches). See [§6](#6-convolutional-neural-networks-with-kerasipynb--convolutional-neural-networks-with-keras).
- **Bias** — a learned per-neuron constant added to the weighted sum before activation, letting a neuron shift its output independent of its inputs. See [§1](#1-forward-propagationipynb--forward-propagation).
- **Chain rule** — the calculus rule that lets a derivative through several nested functions be computed as a product of each function's local derivative; it's the mathematical basis of backpropagation. See [§2](#2-backward-propagationipynb--backward-propagation).
- **Dead neuron** — a ReLU unit whose weighted sum is negative for every training example, so it always outputs 0 and has 0 gradient, permanently stuck. See [§3](#3-activation-functions-and-vanishingipynb--activation-functions--vanishing-gradients).
- **Dropout** — a regularization layer that randomly zeroes a fraction of its input units during training, forcing the network not to over-rely on any single feature. See [§8](#8-classification-and-captioningipynb--capstone-aircraft-damage-classification--captioning).
- **Forward propagation** — computing a network's output by passing inputs through each layer's weighted sum + activation, in order, from input to output. See [§1](#1-forward-propagationipynb--forward-propagation).
- **Gradient descent** — the optimization method that repeatedly nudges each weight in the direction that reduces the loss, scaled by the learning rate. See [§2](#2-backward-propagationipynb--backward-propagation).
- **Learning rate** — the scalar controlling how large each gradient-descent weight update is; too high overshoots, too low trains slowly. See [§2](#2-backward-propagationipynb--backward-propagation).
- **Loss function** — the scalar measure of how wrong a model's predictions are, which training tries to minimize (e.g. MSE for regression, cross-entropy for classification). See [§4](#4-regression-with-kerasipynb--regression-models-with-keras).
- **One-hot encoding** — representing a categorical label as a vector of all zeros except a 1 at the index of the true class, so it can be compared against a softmax output. See [§5](#5-classification-with-kerasipynb--classification-models-with-keras).
- **Overfitting** — when a model fits the training data (including its noise) so closely that it performs worse on new, unseen data. See [§4](#4-regression-with-kerasipynb--regression-models-with-keras).
- **Pooling** — a downsampling operation (typically max pooling) that shrinks a feature map by keeping a summary statistic from each local window. See [§6](#6-convolutional-neural-networks-with-kerasipynb--convolutional-neural-networks-with-keras).
- **Softmax** — an activation that converts a vector of raw scores into a probability distribution over classes that sums to 1. See [§5](#5-classification-with-kerasipynb--classification-models-with-keras).
- **Teacher forcing** — a decoder training technique that feeds in the true previous target token instead of the model's own prior prediction, stabilizing training. See [§7](#7-transformers-with-kerasipynb--seq2seq-with-lstm--attention).
- **Transfer learning** — reusing a model pretrained on one (usually large) dataset as the starting point — often with its layers frozen — for a different, smaller task. See [§8](#8-classification-and-captioningipynb--capstone-aircraft-damage-classification--captioning).
- **Vanishing gradient** — the problem where gradients shrink toward zero as they're multiplied backward through many saturating-activation layers, so early layers barely update. See [§3](#3-activation-functions-and-vanishingipynb--activation-functions--vanishing-gradients).
- **Weighted sum** — the sum of each input multiplied by its corresponding weight, plus a bias; the linear part of a neuron's computation, before activation. See [§1](#1-forward-propagationipynb--forward-propagation).

## Troubleshooting

### Environment & kernel mismatches
- **Wrong kernel selected for `backward-propagation.ipynb`.** It's authored for a `Python (Pyodide)` kernel. If your Jupyter setup doesn't offer that kernel, just select a normal `ipykernel` with `numpy` and `matplotlib` installed (see [Environment Setup](#environment-setup)) — the code itself has nothing Pyodide-specific in it.
- **`ModuleNotFoundError` for `keras` or `tensorflow`.** Confirm you activated the virtual environment created in [Environment Setup](#environment-setup) before launching Jupyter — a kernel started outside that environment won't see the installed packages, and Jupyter's kernel picker can silently default to a different Python than the one you `pip install`ed into.
- **`ImportError` mentioning both `keras` and `tensorflow.keras` disagreeing about a class.** Notebooks 7 and 8 intentionally mix `keras.*` and `tensorflow.keras.*` imports in the same file (see their deep dives) — this is expected and preserved deliberately, not a bug to fix. Make sure you have a `tensorflow-cpu==2.18.0`-compatible `keras` installed (installing the pinned versions from [Environment Setup](#environment-setup) resolves this in practice).
- **`transformers` / `torch` version conflicts.** Install PyTorch with the dedicated CPU-wheel command in [Environment Setup](#environment-setup) (its own `--index-url`) *before or after* the main `pip install` line — installing CPU torch from the default PyPI index can pull in a CUDA build that's far larger and may conflict with `transformers==4.38.2`.

### Training-time expectations
- **`classification-with-keras.ipynb` (notebook 5) is slow.** The notebook's own text warns that its 10-epoch training run "could actually take over 20 minutes" on a CPU-only machine. This is expected, not a hang — reduce `epochs` for a quick sanity check if you just want to confirm the code runs.
- **`transformers-with-Keras.ipynb` (notebook 7) trains 100 epochs.** On the tiny 5-example dataset this is fast in wall-clock time, but if you scale up the dataset size for experimentation, expect training time to grow accordingly.
- **`classification-and-captioning.ipynb` (notebook 8) is the heaviest notebook.** Between VGG16 fine-tuning and BLIP inference, expect the longest total runtime of the set, especially on CPU-only hardware.

### Download failures
- **The capstone's aircraft-damage dataset `.tar` fails to download or extract.** It's fetched at runtime via `urllib.request.urlretrieve` from a hosted URL and extracted with `tarfile`. A partial or interrupted download often leaves a corrupt `.tar` that fails on extraction rather than on download — delete the partially downloaded file and retry rather than re-extracting it.
- **VGG16 `weights='imagenet'` fails to download.** Keras downloads these weights to a local cache (`~/.keras/`) on first use; a flaky connection can leave a corrupt cache entry. Delete the corresponding file under `~/.keras/models/` and retry.
- **BLIP weights (`Salesforce/blip-image-captioning-base`) fail to download from Hugging Face.** These are pulled via `from_pretrained` into the local Hugging Face cache (`~/.cache/huggingface/`). Retry the cell; if it persists, check whether your network blocks `huggingface.co` and consider pre-downloading the weights on a machine that has access.
- **CPU PyTorch wheels fail or time out.** They're hundreds of MB and served from PyTorch's own index (`https://download.pytorch.org/whl/cpu`), not the default PyPI mirror — a slow or filtered connection to that specific host will fail even if regular `pip install` works fine otherwise. Retry with a longer `pip` timeout (`pip install --timeout 120 ...`) if needed.

### Known Notebook Defects

**`classification-with-keras.ipynb` — Practice Exercise 2 `SyntaxError`.**
- *Symptom:* Running the Practice Exercise 2 solution cell raises a `SyntaxError`.
- *Root cause:* The solution cell's final `print(...)` statement is followed by a stray trailing `|` character left over from editing.
- *Fix:* Open the cell and delete the trailing `|` after the closing parenthesis of the last `print(...)` call, then re-run the cell.

**`classification-and-captioning.ipynb` — unset `batch_size` / `n_epochs`.**
- *Symptom:* A configuration cell raises `SyntaxError` on something like `batch_size =` with nothing after the `=`.
- *Root cause:* The cell ships with `batch_size =` and `n_epochs =` present but no values assigned. This cell is *not* one of the 10 numbered Tasks, so it's easy to overlook while hunting through the Task cells for the problem.
- *Fix:* Assign concrete values, e.g. `batch_size = 32` and `n_epochs = 5`, before running the rest of the notebook. Adjust upward once you've confirmed the pipeline works end to end.

**`classification-and-captioning.ipynb` — Task 9 leftover Colab path.**
- *Symptom:* Task 9's caption-generation cell raises `FileNotFoundError` for `/content/sample_image.jpg`.
- *Root cause:* The cell still references a Google Colab-specific absolute path from the notebook's original authoring environment; it was never updated for local/Jupyter use.
- *Fix:* Replace `/content/sample_image.jpg` with the local image path already defined one cell earlier in the notebook (the same test image variable used just before this cell).

**`classification-and-captioning.ipynb` — unused `import zipfile`.**
- *Symptom:* No runtime error; it's just dead code.
- *Root cause:* The dataset is a `.tar` archive extracted with `tarfile`, not a `.zip` — the `zipfile` import appears to be leftover from an earlier version of the notebook.
- *Fix:* None required to run the notebook; safe to delete the import if you're cleaning up your own copy.

## Further Reading

- [Keras documentation](https://keras.io/) — official guides and API reference for everything used in notebooks 4–8.
- [TensorFlow documentation](https://www.tensorflow.org/api_docs) — the backend behind `tensorflow.keras`, used across notebooks 4–8.
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html) — the framework behind BLIP in notebook 8's captioning half.
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index) — the library providing `BlipProcessor` and `BlipForConditionalGeneration` in notebook 8.
- Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (arXiv:1706.03762) — the paper that introduced the scaled dot-product attention mechanism used (in adapted form) in notebook 7.
- Simonyan & Zisserman, ["Very Deep Convolutional Networks for Large-Scale Image Recognition"](https://arxiv.org/abs/1409.1556) (arXiv:1409.1556) — the VGG16 architecture used for transfer learning in notebook 8.
- Li et al., ["BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation"](https://arxiv.org/abs/2201.12086) (arXiv:2201.12086) — the captioning model used in notebook 8.
- [Aircraft Damage Detection dataset on Roboflow Universe](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk) — source of the dent/crack image dataset used in notebook 8, licensed CC BY 4.0.

## License

MIT — see [LICENSE](LICENSE).
