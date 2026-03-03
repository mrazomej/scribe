"""
Compatibility tests for the SVI sampling mixin composition layer.
"""

from scribe.svi._sampling import (
    SamplingMixin,
    _denoise_per_dataset,
    _slice_param_for_dataset,
)
from scribe.svi.results import ScribeSVIResults


def test_sampling_mixin_public_methods_available_on_results():
    """Ensure refactored sampling APIs remain available on results.

    This test guards against accidental API regressions when `SamplingMixin`
    is internally decomposed into smaller sub-mixins.
    """
    expected_methods = [
        "get_posterior_samples",
        "get_predictive_samples",
        "get_ppc_samples",
        "get_map_ppc_samples",
        "get_ppc_samples_biological",
        "get_posterior_ppc_samples",
        "get_map_ppc_samples_biological",
        "denoise_counts_map",
        "denoise_counts_posterior",
        "get_denoised_anndata",
    ]

    for method_name in expected_methods:
        assert hasattr(ScribeSVIResults, method_name)
        assert callable(getattr(ScribeSVIResults, method_name))


def test_sampling_module_compatibility_exports():
    """Ensure historical helper exports remain importable from `_sampling`.

    Keeping these exports stable avoids breaking internal or downstream code
    that imports helpers directly from the historical module path.
    """
    assert SamplingMixin is not None
    assert callable(_slice_param_for_dataset)
    assert callable(_denoise_per_dataset)
