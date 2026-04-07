# Installation

## Using pip

Install the core library:

```bash
pip install scribe
```

Install SCRIBE from source:

```bash
pip install git+https://github.com/mrazomej/scribe.git
```

Install CLI/Hydra tooling:

```bash
pip install "scribe[hydra]"
```

## Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is the recommended package manager for
SCRIBE development:

```bash
git clone https://github.com/mrazomej/scribe.git
cd scribe
uv sync
```

To include CLI/Hydra extras in the active environment:

```bash
uv sync --extra hydra
```

## Development Installation

For contributors and developers:

```bash
git clone https://github.com/mrazomej/scribe.git
cd scribe
uv sync --group dev
```

To include CLI/Hydra tooling during development:

```bash
uv sync --group dev --extra hydra
```

To also install documentation dependencies:

```bash
uv sync --group dev --group docs
```

To install docs + CLI/Hydra tooling together:

```bash
uv sync --group dev --group docs --extra hydra
```

## Breaking Change Note (Hydra Boundary)

- Base `scribe` installs no longer require `hydra-core` or `omegaconf`.
- `scribe-infer` and `scribe-visualize` now require `scribe[hydra]`.
- If a CLI command reports missing optional dependencies, install:
  `pip install "scribe[hydra]"` (or `uv sync --extra hydra`).

## GPU Support

SCRIBE uses [JAX](https://jax.readthedocs.io/en/latest/) for GPU-accelerated
computation. The default installation includes CUDA 12 support. If you are on a
system without a GPU, JAX will automatically fall back to CPU execution.

For specific JAX installation instructions (e.g., different CUDA versions or
TPU support), see the
[JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Docker

A Docker environment is also available for development:

```bash
docker build -t scribe-dev .
docker run --gpus all -it scribe-dev
```

## Verifying Installation

After installation, verify that SCRIBE is working:

```python
import scribe
print(scribe.__doc__)
```

If JAX GPU support is available:

```python
import jax
print(jax.devices())  # Should show GPU device(s)
```
