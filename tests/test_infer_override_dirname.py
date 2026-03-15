"""Tests for compact Hydra override dirname formatting in ``infer.py``."""

from infer import _compact_override_dirname


def test_compact_override_dirname_emits_bare_true_and_omits_false():
    """Encode explicit true as bare key and skip explicit false.

    Returns
    -------
    None
        Asserts the compact formatter emits explicit boolean ``true`` values
        as key-only tokens, omits explicit boolean ``false`` entries, and keeps
        non-boolean key-value entries.
    """
    compact = _compact_override_dirname(
        "mu_dataset_prior=gaussian,gate_dataset_prior=none,"
        "parameterization=mean_odds"
    )

    assert (
        compact
        == "mu_dataset_prior=gaussian,parameterization=mean_odds"
    )


def test_compact_override_dirname_shortens_dot_keys_and_handles_collisions():
    """Shorten dotted keys unless suffix collisions require full key names.

    Returns
    -------
    None
        Asserts unique dotted keys are shortened to their suffix, while
        colliding suffixes are preserved as full dotted keys to avoid
        ambiguity in run directory names.
    """
    compact = _compact_override_dirname(
        "priors.eta_capture=[11.51, 1e-2],foo.eta_capture=7,"
        "inference.batch_size=4096"
    )

    assert (
        compact
        == "priors.eta_capture=11.51,1e-2,foo.eta_capture=7,batch_size=4096"
    )


def test_compact_override_dirname_normalizes_bracket_lists_and_ordering():
    """Normalize bracket lists and move bare true keys first.

    Returns
    -------
    None
        Asserts bracket wrappers and list spacing are removed from values, and
        bare-key boolean tokens are emitted before key-value segments to keep
        parser compatibility with comma-delimited value entries.
    """
    compact = _compact_override_dirname(
        "guide_rank=256,mu_dataset_prior=gaussian,"
        "mixture_params=[phi, mu, gate]"
    )

    assert (
        compact
        == "guide_rank=256,mu_dataset_prior=gaussian,mixture_params=phi,mu,gate"
    )


def test_compact_override_dirname_applies_key_aliases():
    """Apply configured aliases to keys while preserving compact semantics.

    Returns
    -------
    None
        Asserts aliases are applied to keys only and integrated with the
        existing compact formatting rules.
    """
    compact = _compact_override_dirname(
        "mixture_params=[phi, mu, gate],mu_dataset_prior=gaussian",
        aliases={
            "mixture_params": "mixpar",
            "mu_dataset_prior": "mdp",
        },
    )

    assert compact == "mdp=gaussian,mixpar=phi,mu,gate"


def test_compact_override_dirname_keeps_original_key_when_alias_missing():
    """Leave keys unchanged when no alias mapping is provided for them.

    Returns
    -------
    None
        Asserts unknown keys keep their original compact token so alias
        configuration remains optional and non-breaking.
    """
    compact = _compact_override_dirname(
        "guide_rank=256,parameterization=mean_odds",
        aliases={"mixture_params": "mixpar"},
    )

    assert compact == "guide_rank=256,parameterization=mean_odds"


def test_compact_override_dirname_falls_back_on_alias_collisions():
    """Fallback to original tokens when alias expansion collides.

    Returns
    -------
    None
        Asserts colliding alias outputs are replaced by original key tokens to
        avoid ambiguous directory names.
    """
    compact = _compact_override_dirname(
        "mixture_params=phi,mu,gate,joint_params=phi,mu",
        aliases={"mixture_params": "param", "joint_params": "param"},
    )

    assert compact == "mixture_params=phi,mu,gate,joint_params=phi,mu"
