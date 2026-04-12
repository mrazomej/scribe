"""Organism-specific prior defaults for the biology-informed capture model.

Each entry maps an organism name to its expected total mRNA per cell (M_0)
and the log-scale standard deviation (sigma_M) representing cell-to-cell
variation in total mRNA content.  These values are used when
``priors.organism`` is set and explicit ``priors.eta_capture`` is not
provided.

See Also
--------
paper/_capture_prior.qmd : Derivation and justification.
"""

from __future__ import annotations

from typing import Dict, TypedDict


class OrganismPrior(TypedDict):
    """Typed dictionary for organism-specific capture prior parameters.

    Attributes
    ----------
    total_mrna_mean : float
        Expected total mRNA molecules per cell (M_0).
    total_mrna_log_sigma : float
        Log-scale standard deviation of cell-to-cell mRNA variation.
    """

    total_mrna_mean: float
    total_mrna_log_sigma: float


# Lookup table keyed by lowercase organism name.
ORGANISM_PRIORS: Dict[str, OrganismPrior] = {
    "human": {"total_mrna_mean": 200_000, "total_mrna_log_sigma": 0.3},
    "mouse": {"total_mrna_mean": 200_000, "total_mrna_log_sigma": 0.3},
    "yeast": {"total_mrna_mean": 60_000, "total_mrna_log_sigma": 0.3},
    "ecoli": {"total_mrna_mean": 3_000, "total_mrna_log_sigma": 0.3},
}

# Aliases for common alternative names.
ORGANISM_PRIORS["homo_sapiens"] = ORGANISM_PRIORS["human"]
ORGANISM_PRIORS["mus_musculus"] = ORGANISM_PRIORS["mouse"]
ORGANISM_PRIORS["saccharomyces_cerevisiae"] = ORGANISM_PRIORS["yeast"]
ORGANISM_PRIORS["e_coli"] = ORGANISM_PRIORS["ecoli"]
ORGANISM_PRIORS["escherichia_coli"] = ORGANISM_PRIORS["ecoli"]


def resolve_organism_priors(organism: str) -> OrganismPrior:
    """Look up prior parameters for a given organism.

    Parameters
    ----------
    organism : str
        Case-insensitive organism name (e.g. ``"human"``, ``"mouse"``).

    Returns
    -------
    OrganismPrior
        Dictionary with ``total_mrna_mean`` and ``total_mrna_log_sigma``.

    Raises
    ------
    ValueError
        If the organism is not in the lookup table.
    """
    key = organism.strip().lower()
    if key not in ORGANISM_PRIORS:
        supported = sorted(
            k for k in ORGANISM_PRIORS if "_" not in k  # primary names only
        )
        raise ValueError(
            f"Unknown organism '{organism}'. "
            f"Supported organisms: {supported}. "
            "Provide total_mrna_mean and total_mrna_log_sigma directly "
            "for other organisms."
        )
    return ORGANISM_PRIORS[key]
