-- No-op for HTML: the JS-built sidebar TOC reads headings at their
-- original levels, so no shifting is needed for HTML output.
-- For LaTeX/PDF the filter could be extended if necessary.

function Div(el)
  return el
end
