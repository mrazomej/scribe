# tests/models/components

**Purpose.** The neural / distributional building blocks used inside models and
guides: VAE encoders/decoders, covariate embeddings, and normalizing flows.

**Source under test.** `src/scribe/models/components` (`vae_components`,
`covariate_embedding`) and `src/scribe/flows`.

**What lives here.**
- `test_vae_components` — VAE encoder/decoder Flax Linen components.
- `test_covariate_embedding` — covariate embeddings (`CovariateSpec`) and the flow stacks they compose with.
- `test_flows` — the normalizing-flows module (autoregressive/coupling transforms, flow chains).

**What does NOT live here.**
- The VAE *results* class (`svi.vae_results`) → `tests/inference/` (`test_vae_results`).
- Full VAE pipeline (encode→decode→sample end-to-end) → `tests/integration/` (`test_vae_integration`).
- Latent-spec / latent-dispatch → `../builders/` (`test_latent_spec`) and `tests/inference/` (`test_latent_dispatch`).

**Key fixtures.** Root `tests/conftest.py`. No folder-local conftest.
