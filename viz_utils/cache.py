"""Cache path helpers for UMAP and other persisted data."""

import os
import re
import hashlib


def _sanitize_cache_token(value, max_len=48):
    """Create a filesystem-friendly token for cache filenames."""
    as_str = str(value).strip()
    if as_str == "":
        return "none"
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", as_str)
    token = token.strip("._-")
    if token == "":
        token = "none"
    if len(token) > max_len:
        token = token[:max_len]
    return token


def _build_umap_cache_path(cfg, cache_umap):
    """Build a split-aware path for persisted UMAP reducer caches."""
    if not cache_umap:
        return None

    data_cfg = cfg.data if hasattr(cfg, "data") else None
    data_path = data_cfg.path if data_cfg is not None else None
    if not data_path:
        return None

    data_dir = os.path.dirname(data_path)
    data_stem = os.path.splitext(os.path.basename(data_path))[0]
    subset_column = data_cfg.get("subset_column", None)
    subset_value = data_cfg.get("subset_value", None)

    if subset_column is not None and subset_value is not None:
        subset_col_token = _sanitize_cache_token(subset_column)
        subset_val_token = _sanitize_cache_token(subset_value)
        cache_identity = (
            f"{os.path.abspath(data_path)}|{subset_column}|{subset_value}"
        )
        cache_hash = hashlib.sha1(cache_identity.encode("utf-8")).hexdigest()[
            :12
        ]
        cache_name = (
            f"{data_stem}_umap_{subset_col_token}_{subset_val_token}_"
            f"{cache_hash}.pkl"
        )
    else:
        cache_name = f"{data_stem}_umap_full.pkl"

    return os.path.join(data_dir, cache_name)
