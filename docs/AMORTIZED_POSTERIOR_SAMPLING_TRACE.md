# Trace: Amortized Capture Probability Posterior Sampling

This document traces the complete flow of how posterior samples are generated for an amortized capture probability parameter, from data input to final posterior samples.

## Overview

For amortized models, the capture probability (`phi_capture` or `p_capture`) is not learned as separate parameters per cell. Instead, a neural network (amortizer) predicts variational distribution parameters from the data, and we sample from those distributions.

## Complete Flow

### Step 1: User Calls `get_posterior_samples()`
**Location**: `src/scribe/svi/_sampling.py:30-96`

```python
posterior_samples = results.get_posterior_samples(
    rng_key=rng_key,
    n_samples=100,
    counts=counts  # Shape: (n_cells, n_genes) - FULL counts matrix
)
```

**What happens**:
- Gets the guide function via `self._model_and_guide()`
- Prepares `model_args` with `n_cells`, `n_genes`, `model_config`
- Calls `sample_variational_posterior()` with `counts` parameter

**Key point**: The FULL `counts` matrix is passed (not subsetted by genes), because the amortizer needs total UMI count per cell.

---

### Step 2: `sample_variational_posterior()` Sets Up Predictive
**Location**: `src/scribe/sampling.py:17-80`

```python
def sample_variational_posterior(
    guide: Callable,
    params: Dict,  # Optimized variational parameters (includes amortizer weights)
    model: Callable,
    model_args: Dict,
    rng_key: random.PRNGKey,
    n_samples: int = 100,
    counts: Optional[jnp.ndarray] = None,
) -> Dict:
```

**What happens**:
1. **Line 59-61**: Adds `counts` to `model_args` if provided
   ```python
   if counts is not None:
       model_args = {**model_args, "counts": counts}
   ```

2. **Line 64**: Creates `Predictive` object with the guide
   ```python
   predictive_param = Predictive(guide, params=params, num_samples=n_samples)
   ```
   - `params` contains the optimized amortizer network weights
   - `num_samples=n_samples` means we'll generate `n_samples` independent samples

3. **Line 67**: Runs the guide through `Predictive`
   ```python
   posterior_samples = predictive_param(rng_key, **model_args)
   ```
   - This calls `guide(n_cells, n_genes, model_config, counts=counts)` `n_samples` times
   - Each call generates one sample from the variational posterior
   - Results are stacked: shape becomes `(n_samples, ...)` for each parameter

---

### Step 3: Guide Function Executes
**Location**: `src/scribe/models/builders/guide_builder.py:1106-1233`

The guide function is built by `GuideBuilder.build()`. For cell-specific parameters with amortization:

```python
def guide(
    n_cells: int,
    n_genes: int,
    model_config: "ModelConfig",
    counts: Optional[jnp.ndarray] = None,  # Shape: (n_cells, n_genes)
    batch_size: Optional[int] = None,
):
```

**What happens**:
1. Sets up dimensions dict
2. Sets up guides for global and gene-specific parameters (standard variational params)
3. **Lines 1202-1231**: Sets up guides for **cell-specific parameters** inside a cell plate:
   ```python
   cell_specs = [s for s in specs if s.is_cell_specific]
   if cell_specs:
       with numpyro.plate("cells", n_cells):
           for spec in cell_specs:
               guide_family = spec.guide_family or MeanFieldGuide()
               setup_cell_specific_guide(
                   spec,
                   guide_family,  # This is AmortizedGuide
                   dims,
                   model_config,
                   counts=counts,  # Full counts matrix passed here
                   batch_idx=None,  # Full sampling mode
               )
   ```

---

### Step 4: `setup_cell_specific_guide()` for AmortizedGuide
**Location**: `src/scribe/models/builders/guide_builder.py:710-761` (for BetaPrimeSpec)

For `phi_capture` with `BetaPrimeSpec` and `AmortizedGuide`:

```python
@dispatch(BetaPrimeSpec, AmortizedGuide, dict, object)
def setup_cell_specific_guide(
    spec: BetaPrimeSpec,
    guide: AmortizedGuide,
    dims: Dict[str, int],
    model_config: "ModelConfig",
    counts: Optional[jnp.ndarray] = None,
    batch_idx: Optional[jnp.ndarray] = None,
    **kwargs,
) -> jnp.ndarray:
```

**What happens**:

1. **Line 750-751**: Validates counts are provided
   ```python
   if counts is None:
       raise ValueError("Amortized guide requires counts data")
   ```

2. **Line 754**: Gets data for current batch
   ```python
   data = counts if batch_idx is None else counts[batch_idx]
   ```
   - If `batch_idx is None` (full sampling): `data = counts` → shape `(n_cells, n_genes)`
   - If batching: `data = counts[batch_idx]` → shape `(batch_size, n_genes)`

3. **Line 757**: **Amortizer predicts variational parameters**
   ```python
   var_params = guide.amortizer(data)
   ```
   - This is the key step! The amortizer network processes the counts data
   - Returns a dict like `{"alpha": array, "beta": array}`
   - The amortizer's output_transforms (softplus+offset+clamp) ensure positive values

4. **Lines 758-759**: Uses the already-positive parameters directly
   ```python
   alpha = var_params["alpha"]  # Shape: (n_cells,), already positive
   beta = var_params["beta"]    # Shape: (n_cells,), already positive
   ```

5. **Line 761**: Samples from the variational distribution
   ```python
   return numpyro.sample(spec.name, BetaPrime(alpha, beta))
   ```
   - `spec.name` is `"phi_capture"`
   - Creates a `BetaPrime` distribution with per-cell parameters `(alpha[i], beta[i])`
   - Samples one value per cell: shape `(n_cells,)`

---

### Step 5: Amortizer Network Forward Pass
**Location**: `src/scribe/models/components/amortizers.py:280-315`

When `guide.amortizer(data)` is called:

```python
def __call__(self, data: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Forward pass through the amortizer network."""
```

**What happens**:

1. **Line 295**: Compute sufficient statistic from data
   ```python
   h = self.sufficient_statistic.compute(data)
   ```
   - For `TOTAL_COUNT`: `h = jnp.log1p(data.sum(axis=-1, keepdims=True))`
   - Input: `data` shape `(n_cells, n_genes)`
   - Output: `h` shape `(n_cells, 1)` - one scalar per cell (log1p of total UMI)

2. **Lines 300-301**: Forward through hidden layers
   ```python
   for layer in self.layers:
       h = jax.nn.relu(layer(h))
   ```
   - `h` shape: `(n_cells, 1)` → `(n_cells, 64)` → `(n_cells, 32)`
   - Each layer: `Linear(input_dim, output_dim)` followed by ReLU

3. **Lines 308-313**: Compute outputs from each head
   ```python
   for name, head in self.output_heads.items():
       out = head(h).squeeze(-1)  # Shape: (n_cells,)
       if name in self.output_transforms:
           out = self.output_transforms[name](out)
       outputs[name] = out
   ```
   - `h` shape: `(n_cells, 32)`
   - Each output head: `Linear(32, 1)` → squeeze → shape `(n_cells,)`
   - Returns: `{"alpha": (n_cells,), "beta": (n_cells,)}`

**Key transformation**:
```
counts (n_cells, n_genes)
  ↓ sum(axis=-1)
total_umis (n_cells,)
  ↓ log1p
log1p_total (n_cells, 1)
  ↓ MLP [64, 32]
hidden (n_cells, 32)
  ↓ output_heads + output_transforms (softplus+offset+clamp)
{alpha: (n_cells,), beta: (n_cells,)}  — already positive
  ↓ BetaPrime(alpha, beta).sample()
phi_capture (n_cells,)
```

---

### Step 6: NumPyro Predictive Collects Samples
**Location**: NumPyro's `Predictive` class (external)

When `predictive_param(rng_key, **model_args)` is called:

1. **For each sample `i` in `range(n_samples)`**:
   - Calls `guide(n_cells, n_genes, model_config, counts=counts)` with a different `rng_key`
   - Inside the guide, `numpyro.sample("phi_capture", ...)` is called
   - This samples `phi_capture` with shape `(n_cells,)` for this sample

2. **Stacks all samples**:
   - Sample 0: `phi_capture[0]` shape `(n_cells,)`
   - Sample 1: `phi_capture[1]` shape `(n_cells,)`
   - ...
   - Sample n-1: `phi_capture[n-1]` shape `(n_cells,)`
   - **Result**: `posterior_samples["phi_capture"]` shape `(n_samples, n_cells)`

---

## Summary: Data Flow

```
User provides:
  counts: (n_cells, n_genes) - FULL count matrix

↓ get_posterior_samples()

↓ sample_variational_posterior()
  - Creates Predictive(guide, params, num_samples=n_samples)
  - params contains amortizer network weights

↓ Predictive calls guide() n_samples times

↓ guide() sets up cell plate
  - with numpyro.plate("cells", n_cells):
      setup_cell_specific_guide(..., counts=counts, ...)

↓ setup_cell_specific_guide() for AmortizedGuide
  - data = counts  # (n_cells, n_genes)
  - var_params = guide.amortizer(data)

↓ amortizer.__call__(data)
  - h = log1p(sum(data, axis=-1))  # (n_cells, 1) - total UMI per cell
  - h = MLP(h)  # (n_cells, 1) → (n_cells, 64) → (n_cells, 32)
  - alpha, beta = output_heads(h) + output_transforms  # each: (n_cells,), positive
  - return {"alpha": alpha, "beta": beta}

↓ setup_cell_specific_guide() continues
  - alpha = var_params["alpha"]  # (n_cells,), already positive
  - beta = var_params["beta"]    # (n_cells,), already positive
  - phi_capture = numpyro.sample("phi_capture", BetaPrime(alpha, beta))
    # Samples (n_cells,) values, one per cell

↓ Predictive collects n_samples
  - posterior_samples["phi_capture"] shape: (n_samples, n_cells)
```

## Key Points

1. **Full counts matrix is used**: The amortizer needs total UMI count per cell, which requires summing across ALL genes. Subsetting by genes would give wrong totals.

2. **Per-cell computation**: The amortizer processes all cells in parallel:
   - Input: `(n_cells, n_genes)` → computes `(n_cells, 1)` sufficient statistics
   - MLP processes all cells at once: `(n_cells, 1)` → `(n_cells, 32)`
   - Output: `(n_cells,)` variational parameters per output head

3. **Sampling happens inside plate**: `numpyro.sample("phi_capture", ...)` is called inside `numpyro.plate("cells", n_cells)`, which samples one value per cell.

4. **Predictive stacks samples**: NumPyro's `Predictive` calls the guide `n_samples` times and stacks results, giving final shape `(n_samples, n_cells)`.

5. **Network weights are in params**: The amortizer network weights are part of the optimized `params` dictionary, so they're learned during SVI training.

## Why This Works

- **Sufficient statistic**: Total UMI count is theoretically sufficient for capture probability in the NB-DM model
- **Amortization**: Instead of learning `n_cells` separate parameters, we learn a small network that predicts parameters from data
- **Scalability**: Works for any number of cells without increasing the number of variational parameters
- **Sharing**: Similar cells (with similar total UMIs) get similar variational parameters, sharing statistical strength