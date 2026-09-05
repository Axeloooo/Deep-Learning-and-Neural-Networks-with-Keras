# Contributing

Thank you for helping improve this collection of deep-learning lessons. Contributions can include mathematical corrections, clearer explanations, notebook fixes, better exercises, or documentation improvements. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before you start

Read the [README](README.md) for the learning path and supported Google Colab workflow. Check existing [issues](https://github.com/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras/issues) and [pull requests](https://github.com/Axeloooo/Deep-Learning-and-Neural-Networks-with-Keras/pulls) to avoid duplicating work. For substantial changes or new lessons, open an issue describing the learning objective and proposed scope first. Small corrections can go directly into a pull request.

Each notebook is a standalone lesson. Keep contributions focused enough that a reviewer can understand the educational purpose and reproduce the result. Coding agents should follow [AGENTS.md](AGENTS.md).

## Reporting a problem

Include the notebook filename and affected cell heading, the steps needed to reproduce the problem, expected and actual behavior, and the relevant error text. For runtime problems, include whether you used Colab with CPU or GPU, the Python version, and the versions of the affected libraries. For a mathematical or explanatory error, identify the expression or passage and explain the correction.

Remove credentials, personal information, and private data from reports and notebook outputs. Conduct concerns should follow the reporting process in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Making a change

1. Fork the repository and create a branch for your change.
2. Edit the relevant Markdown file or notebook. In Colab, work on a copy and download the edited `.ipynb` into the matching path in your fork before committing.
3. Validate the change using the guidance below.
4. Review the diff for unintended changes, then commit using a Conventional Commit message.
5. Open a pull request targeting `main`, explaining the problem, the change, and how you checked it. Link a related issue when applicable.

## Notebook conventions

- Keep each notebook runnable independently, with imports and setup before the cells that need them.
- Use the existing `Python 3 (ipykernel)` kernel metadata. Preserve the surrounding `keras.*` or `tensorflow.keras.*` import style.
- Leave the commented installation cells commented out. Colab is the supported environment; this repository does not maintain a local dependency manifest.
- Explain the purpose of each experiment, important array shapes, and how to interpret its results. Keep Markdown explanations consistent with the code.
- Preserve practice exercises and existing source acknowledgments. Attribute material you add and ensure it can be included in the repository.
- Keep useful instructional outputs, but avoid unrelated execution-count, output, or metadata changes. Remove debug output and any sensitive values before submitting.
- Do not commit downloaded datasets, model weights, generated `.keras` or `.h5` files, checkpoints, or local environments. Keep download and preparation steps inside the lesson when needed.
- Update the README and agent notebook map if you add, rename, or materially change a lesson.

## Validation

There is no application build, lint command, or unit-test suite. The release workflow does not check notebook execution.

For a notebook change, open the edited version in a fresh Colab runtime and execute its cells in order. Inspect results and plots as well as checking for exceptions: a cell can run successfully and still teach an incorrect calculation. Confirm that the lesson runs without hidden state from an earlier session and that its explanations match the observed results. Intentional exercise placeholders may remain unanswered.

In the pull request, record the notebook tested, runtime type, relevant library versions, and outcome. Note any cells you could not run and why. Do not claim full execution when only selected cells were checked. If downloads or runtime availability prevent validation, report that limitation explicitly.

For Markdown-only changes, check relative links, table-of-contents anchors, notebook filename casing, and consistency with the actual repository. Notebook execution is not required for prose-only edits; changes to executable examples or setup instructions need the corresponding execution check.

Before submitting, run:

```sh
git diff --check
```

This checks whitespace in the diff; it does not validate notebook behavior. Inspect newly added files too, since untracked files are not included in `git diff` until staged.

## Commits and releases

Use Conventional Commits, for example:

```text
docs: clarify the backpropagation learning path
fix: correct the sigmoid derivative example
feat: add an activation comparison exercise
```

Use `docs` for documentation, `fix` for corrections, and `feat` for new functionality or lesson content. Describe breaking changes with the Conventional Commits breaking-change notation when applicable.

The [release workflow](.github/workflows/release.yaml) runs on pushes to `main` and supports manual dispatch. It uses [`.releaserc`](.releaserc) to generate version tags, release notes, and GitHub releases. The configuration explicitly treats `docs`, `refactor`, and `perf` commits as patch releases. There is no package to build or publish, and contributors should not add the workflow's generated npm files to the repository.

## Pull request checklist

- The change has a clear learning or maintenance purpose and stays within its stated scope.
- Explanations, notebook names, and documentation links agree with the implementation.
- Relevant validation and any limitations are recorded in the pull request.
- The diff contains no accidental notebook rewrites, downloaded assets, or sensitive information.
- Commit messages follow Conventional Commits.
