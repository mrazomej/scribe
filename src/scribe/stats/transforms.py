"""Custom NumPyro Transform subclasses.

Mirrors the role of :mod:`scribe.stats.distributions` for transforms:
this module is the canonical home for any ``numpyro.distributions.transforms.Transform``
subclass scribe defines. Keep this module leaf-clean (numpyro / jax only,
no scribe-internal imports) so that downstream packages like
``scribe.stats.jacobian_map`` can dispatch on these classes at module
load without dragging in heavyweight subtrees.

Contents
--------
SlicedTransform
    Applies different bijective transforms to contiguous slices of a
    vector. The Jacobian is block-diagonal across slices, so the
    log-abs-det is the sum of per-slice contributions. Used by the
    concatenated joint flow guide (see
    :mod:`scribe.models.builders._guide_joint_flow_mixin`) and by the
    per-slice MAP recursion in :func:`scribe.stats.jacobian_corrected_map`.

Backward compatibility
----------------------
For historical reasons ``SlicedTransform`` was previously defined in
``scribe.flows.distributions`` and exported via ``scribe.flows.__init__``.
Both of those import paths still work via re-export, so existing user
code is unaffected.
"""

from __future__ import annotations

from typing import List

import jax.numpy as jnp
from numpyro.distributions import constraints
from numpyro.distributions.transforms import Transform


class SlicedTransform(Transform):
    """Apply different bijective transforms to contiguous slices of a vector.

    Given a vector of dimension ``D = sum(sizes)``, this transform splits it
    into ``len(sizes)`` contiguous slices and applies a per-slice transform.
    The forward pass concatenates the transformed slices back into a ``D``-dim
    vector; the inverse reverses the process.

    The Jacobian is block-diagonal (each slice is independent), so
    ``log_abs_det_jacobian`` is the sum of per-slice contributions.

    This is the key building block for the concatenated joint flow guide: the
    base distribution is a ``FlowDistribution`` over ``R^D``, and wrapping it in
    ``TransformedDistribution(flow_dist, SlicedTransform(...))`` yields a
    distribution whose samples live in the product of per-spec constrained
    spaces with a correct ELBO.

    Parameters
    ----------
    transforms : list of Transform
        One transform per slice.  Element-wise transforms (Exp,
        Sigmoid, Softplus, etc.) are expected.
    sizes : list of int
        Trailing-dimension sizes of each slice.
        ``sum(sizes)`` must equal the vector dimension.

    Examples
    --------
    >>> from numpyro.distributions.transforms import ExpTransform, SigmoidTransform
    >>> t = SlicedTransform(
    ...     transforms=[SigmoidTransform(), ExpTransform()],
    ...     sizes=[1, 20],
    ... )
    >>> y = t(jnp.zeros(21))  # first elem -> sigmoid, rest -> exp
    """

    def __init__(
        self,
        transforms: List[Transform],
        sizes: List[int],
    ):
        if len(transforms) != len(sizes):
            raise ValueError(
                f"len(transforms)={len(transforms)} != len(sizes)={len(sizes)}"
            )
        self._transforms = list(transforms)
        self._sizes = list(sizes)

        # Pre-compute cumulative offsets for slicing
        self._offsets: List[int] = []
        running = 0
        for s in sizes:
            self._offsets.append(running)
            running += s

    # -- NumPyro Transform protocol ------------------------------------

    @property
    def domain(self):
        return constraints.real_vector

    @property
    def codomain(self):
        return constraints.real_vector

    def __call__(self, x):
        pieces = []
        for i, t in enumerate(self._transforms):
            start = self._offsets[i]
            end = start + self._sizes[i]
            pieces.append(t(x[..., start:end]))
        return jnp.concatenate(pieces, axis=-1)

    def _inverse(self, y):
        pieces = []
        for i, t in enumerate(self._transforms):
            start = self._offsets[i]
            end = start + self._sizes[i]
            pieces.append(t._inverse(y[..., start:end]))
        return jnp.concatenate(pieces, axis=-1)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        total = jnp.zeros(x.shape[:-1])
        for i, t in enumerate(self._transforms):
            start = self._offsets[i]
            end = start + self._sizes[i]
            x_i = x[..., start:end]
            y_i = y[..., start:end]
            # Element-wise transforms return per-element ladj; sum over the
            # slice dimension to get the scalar contribution for this block.
            ladj_i = t.log_abs_det_jacobian(x_i, y_i)
            total = total + jnp.sum(ladj_i, axis=-1)
        return total

    def tree_flatten(self):
        return (self._transforms, self._sizes), (
            ("_transforms", "_sizes"),
            dict(),
        )

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        transforms, sizes = params
        return cls(transforms=transforms, sizes=sizes)


__all__ = ["SlicedTransform"]
