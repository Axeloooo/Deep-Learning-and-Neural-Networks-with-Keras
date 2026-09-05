# AGENTS.md

Repository guidance for Codex and other coding agents.

## Scope and documentation

- Keep changes focused on the requested lesson or documentation task; preserve unrelated work in the working tree.
- The notebooks are an evolving educational collection. Describe implemented examples accurately without claiming that planned improvements are complete.
- `README.md` is the learner entry point, `CONTRIBUTING.md` describes the contribution and validation workflow, and `CODE_OF_CONDUCT.md` defines community expectations. Keep these documents consistent when changing notebook names, scope, or setup.
- Do not recreate `CLAUDE.md` or Claude-specific configuration as part of routine work.

## Repository purpose

This is an educational collection of standalone Jupyter notebooks covering deep learning and neural network fundamentals, built with Keras/TensorFlow (and some PyTorch/Hugging Face Transformers for one notebook). There is no application code, package, build system, or test suite — each notebook under `notebooks/` is a self-contained lesson that can be run independently in a Jupyter environment.

## Working with notebooks

- No `requirements.txt` / `environment.yml` exists in the repo, and the README documents Google Colab as the only supported way to run the notebooks. Notebooks assume `keras`, `tensorflow`, `numpy`, `matplotlib`, `pandas`, and (for `classification-and-captioning.ipynb` only) `torch`, `transformers`, and `Pillow` are available — the workflow relies on Colab's provided environment. Runtime packages can change; report compatibility failures with the actual runtime and library versions rather than claiming an untested version range is supported. The notebooks' own `!pip install` cells are all commented out on purpose; do not re-enable them.
- There is no lint, test, or build command — validate changes by executing the notebook's cells (e.g. via `jupyter nbconvert --to notebook --execute <file>` or running it in Jupyter/JupyterLab) rather than by writing unit tests.
- For documentation-only changes, check Markdown links, heading anchors, notebook paths, and `git diff --check`; notebook execution is unnecessary unless runnable instructions or code examples change.
- For notebook changes, run from a fresh runtime in cell order, inspect numerical results and plots, and report any cells that could not be validated. Preserve intentional exercise placeholders, existing attribution, and useful instructional output; avoid unrelated output or metadata churn.
- Do not commit downloaded datasets, model weights, generated models, notebook checkpoints, or local environment files.
- All eight notebooks use a standard `Python 3 (ipykernel)` kernel.
- Notebooks import from `keras.*` directly in some places and `tensorflow.keras.*` in others (even within the same notebook, e.g. `transformers-with-Keras.ipynb` and `classification-and-captioning.ipynb`) — preserve whichever import style the surrounding cells already use rather than normalizing it.

## Notebook map

| Notebook                                         | Topic                                                                                                           |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `forward-propagation.ipynb`                      | Forward propagation from scratch with NumPy                                                                     |
| `backward-propagation.ipynb`                     | Backward propagation from scratch with NumPy                                                                    |
| `activation-functions-and-vanishing.ipynb`       | Activation functions and the vanishing gradient problem                                                         |
| `regression-with-keras.ipynb`                    | Regression models with Keras `Sequential`                                                                       |
| `classification-with-keras.ipynb`                | Classification on MNIST with Keras `Sequential`                                                                 |
| `convolutional-neural-networks-with-keras.ipynb` | CNNs (Conv2D/MaxPooling2D) on MNIST                                                                             |
| `transformers-with-Keras.ipynb`                  | English–Spanish seq2seq with LSTM + cross-attention (Keras/TF); not a full Transformer                          |
| `classification-and-captioning.ipynb`            | Transfer learning with VGG16 for damage classification, plus BLIP-based image captioning (Transformers/PyTorch) |

## Release process

Releases are automated via `semantic-release` (config in `.releaserc`), triggered by `.github/workflows/release.yaml` on pushes to `main`. Commit messages must follow Conventional Commits format, since `@semantic-release/commit-analyzer` derives version bumps from them. There is no `package.json` checked into the repo — the workflow runs `npm init -y` and installs `semantic-release` fresh on each run solely to drive versioning/release notes/GitHub releases; it does not build or publish a package.

The workflow also supports manual dispatch. The release configuration explicitly assigns patch releases to `docs`, `refactor`, and `perf` commits. The workflow performs release automation, not notebook validation.
