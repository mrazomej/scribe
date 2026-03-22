# Contributing

Thank you for your interest in contributing to SCRIBE! We welcome contributions
from the community and are grateful for any help you can provide.

For the full contributing guide (code of conduct, development process, PR
checklist, etc.), see
[CONTRIBUTING.md](https://github.com/mrazomej/scribe/blob/main/CONTRIBUTING.md)
on GitHub.

## Building the Documentation Locally

The documentation is built with [MkDocs](https://www.mkdocs.org/) and the
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

### Setup

Install the documentation dependencies:

```bash
uv sync --only-group docs
```

### Live Preview

Start a local development server with live-reload:

```bash
uv run mkdocs serve
```

Then open <http://127.0.0.1:8000> in your browser. Changes to any Markdown
file will be reflected instantly.

### Building

To build the static site:

```bash
uv run mkdocs build --strict
```

The `--strict` flag treats warnings as errors, ensuring all links are valid
and no pages are missing.

## Documentation Style Guide

- Use **Markdown** for all documentation pages
- Write **NumPy-style docstrings** in Python source code — they are
  automatically rendered in the API reference
- Use `\(...\)` for inline math and `\[...\]` for display math (MathJax)
- Use [admonitions](https://squidfunk.github.io/mkdocs-material/reference/admonitions/)
  (`!!! note`, `!!! warning`, `!!! tip`) for callouts
- Use [content tabs](https://squidfunk.github.io/mkdocs-material/reference/content-tabs/)
  (`=== "Tab Name"`) for model-specific code examples
- Keep sentences concise and technical — this is a scientific package
