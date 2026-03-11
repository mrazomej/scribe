# `viz_utils` Notes

## PPC Performance Controls

PPC plotting now includes adaptive histogram capping and adaptive rendering for
high-count genes. These options are configured under `viz.ppc_opts` in
`conf/viz/default.yaml` and are shared by standard PPC, bio-PPC, mixture PPC,
and annotation PPC.

- `n_samples` (default `512`): number of predictive draws used in PPC panels.
- `hist_max_bin_quantile` (default `0.99`): quantile of observed counts used to
  cap histogram bins before credible-region computation.
- `hist_max_bin_floor` (default `10`): minimum bin cap so low-count genes still
  display a useful range.
- `render_auto_line_bin_threshold` (default `1000`): if plotted bins exceed
  this threshold, PPC bands switch from `stairs` to line/fill rendering.
- `render_line_target_points` (default `200`): target x-points for decimated
  line-mode rendering.
- `render_line_interpolate` (default `true`): interpolate line-mode curves
  (`true`) or use nearest-bin decimation (`false`).

## Behavior Summary

- Small/medium count ranges stay visually identical to previous `stairs` plots.
- Large count ranges avoid expensive full-resolution rendering by switching to
  compact line-mode plotting after bin capping.
- Histogram credible regions are now capped before percentile computation, which
  significantly reduces compute time for heavy-tailed/high-expression genes.
