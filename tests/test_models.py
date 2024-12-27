"""
Unit tests for model definitions and functionality.
"""

import pytest
from scribe.models import (
    nbdm_model, nbdm_guide,
    zinb_model, zinb_guide,
    nbvcp_model, nbvcp_guide,
    zinbvcp_model, zinbvcp_guide,
    get_model_and_guide
)

def test_get_model_and_guide():
    """Test model registry functionality."""
    # Test valid model types
    model, guide = get_model_and_guide("nbdm")
    assert model == nbdm_model
    assert guide == nbdm_guide
    
    model, guide = get_model_and_guide("zinb")
    assert model == zinb_model
    assert guide == zinb_guide

    model, guide = get_model_and_guide("nbvcp")
    assert model == nbvcp_model
    assert guide == nbvcp_guide
    
    model, guide = get_model_and_guide("zinbvcp")
    assert model == zinbvcp_model
    assert guide == zinbvcp_guide
    
    # Test invalid model type
    with pytest.raises(ValueError):
        get_model_and_guide("invalid_model")