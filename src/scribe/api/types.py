"""
Return-type alias for :func:`scribe.api.fit`.
"""

from typing import Union

from ..svi.results import ScribeSVIResults
from ..mcmc.results import ScribeMCMCResults
from ..svi.vae_results import ScribeVAEResults

# Type alias covering all result objects that ``fit`` may return.
ScribeResults = Union[ScribeSVIResults, ScribeMCMCResults, ScribeVAEResults]
