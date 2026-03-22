# Installation

## Using pip

SCRIBE can be installed from source:

```bash
pip install git+https://github.com/mrazomej/scribe.git
```

## Using uv (recommended)

[uv](https://docs.astral.sh/uv/) is the recommended package manager for
SCRIBE development:

```bash
git clone https://github.com/mrazomej/scribe.git
cd scribe
uv sync
```

## Development Installation

For contributors and developers:

```bash
git clone https://github.com/mrazomej/scribe.git
cd scribe
uv sync --group dev
```

To also install documentation dependencies:

```bash
uv sync --group dev --group docs
```

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
