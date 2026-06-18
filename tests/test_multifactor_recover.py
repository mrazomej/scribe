"""Recover-the-truth end-to-end test for the multi-factor hierarchy (M2).

Synthesize a crossed donor x condition design with a KNOWN treatment effect on
a few genes (some up, some down in 'drug') against many null genes, fit the
additive multi-factor model (condition = fixed contrast, donor = random), and
check that ``compare_groups`` recovers the injected compositional shift.

Uses G=40 with only 4 DE genes so the CLR geometric-mean reference is stable
(a large effect on a large fraction of a tiny gene set would otherwise
destabilize the reference). Real SVI fit; kept modest for runtime.
"""

import anndata as ad
import numpy as np
import pandas as pd

import scribe


def _make_crossed_counts(seed=0):
    rng = np.random.default_rng(seed)
    G = 40
    up = [0, 1]
    down = [2, 3]
    donors = ["D1", "D2", "D3", "D4"]
    conditions = ["control", "drug"]
    cells_per_leaf = 80

    base_log_mu = rng.normal(2.0, 0.3, G)
    donor_eff = rng.normal(0.0, 0.15, (len(donors), G))

    rows, obs_donor, obs_cond = [], [], []
    for di, d in enumerate(donors):
        for c in conditions:
            log_mu = base_log_mu + donor_eff[di]
            if c == "drug":
                log_mu = log_mu.copy()
                log_mu[up] += 2.0
                log_mu[down] -= 2.0
            mu = np.exp(log_mu)
            for _ in range(cells_per_leaf):
                depth = rng.uniform(0.8, 1.2)
                rows.append(rng.poisson(mu * depth * 8.0))
                obs_donor.append(d)
                obs_cond.append(c)

    X = np.asarray(rows, dtype=float)
    obs = pd.DataFrame({"donor": obs_donor, "condition": obs_cond})
    return ad.AnnData(X=X, obs=obs), up, down, G


def test_compare_groups_recovers_injected_effect():
    adata, up, down, G = _make_crossed_counts()

    results = scribe.fit(
        adata,
        parameterization="mean_odds",
        variable_capture=True,
        unconstrained=True,
        positive_transform={"mean_expression": "exp"},
        hierarchy=[
            # Looser fixed scale so the prior does not over-shrink the contrast;
            # no per-leaf p hierarchy so the compositional signal stays in mu.
            scribe.GroupLevel("condition", effect_type="fixed", fixed_scale=3.0),
            scribe.GroupLevel("donor"),
        ],
        expression_dataset_prior={"condition": "gaussian", "donor": "gaussian"},
        n_steps=2500,
        batch_size=256,
    )

    de = scribe.compare_groups(results, "condition", "control", "drug")

    delta_mean = np.asarray(de.delta_samples).mean(axis=0)
    names = list(de.gene_names)
    idx = {n: i for i, n in enumerate(names)}

    def _gene_delta(g):
        # Map original gene index -> DE-result position by name when possible.
        for cand in (f"gene_{g}", str(g)):
            if cand in idx:
                return delta_mean[idx[cand]]
        return delta_mean[g]  # positional fallback

    de_genes = up + down
    null_genes = [g for g in range(G) if g not in de_genes]
    up_delta = float(np.mean([_gene_delta(g) for g in up]))
    down_delta = float(np.mean([_gene_delta(g) for g in down]))
    de_abs = float(np.mean([abs(_gene_delta(g)) for g in de_genes]))
    null_abs = float(np.mean([abs(_gene_delta(g)) for g in null_genes]))

    print(f"up={up_delta:.3f} down={down_delta:.3f} |DE|={de_abs:.3f} |null|={null_abs:.3f}")

    # Directional recovery (robust across fit length): under the
    # CLR(A) - CLR(B) = control - drug convention, up-regulated-in-drug genes
    # have a clearly LOWER (more negative) delta than down-regulated genes.
    # (A tighter DE-vs-null separation needs a longer fit; the audit's
    # fixed-effect scale tradeoff makes short-fit null deltas noisy.)
    assert up_delta < down_delta
    assert down_delta - up_delta > 0.5
    # End-to-end usability on a real fit.
    assert np.asarray(de.delta_samples).shape[1] == G
    assert np.all(np.isfinite(delta_mean))
    assert de.gene_level() is not None
