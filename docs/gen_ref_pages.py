"""Generate the code reference pages for the SCRIBE API documentation.

This script is used by mkdocs-gen-files to automatically create one
documentation page per Python module found under src/scribe/. The generated
pages use mkdocstrings ::: directives so that docstrings are rendered at
build time without needing to import the package.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
mod_symbol = '<code class="doc-symbol doc-symbol-nav doc-symbol-module"></code>'

# Walk every .py file under src/scribe/
src = Path("src")

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    # Skip private modules (but keep __init__)
    if parts[-1].startswith("_") and parts[-1] != "__init__":
        continue

    # Skip legacy modules (deprecated, not part of the public API)
    if "legacy" in parts:
        continue

    # Turn __init__ into section index pages
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    if not parts:
        continue

    # Build navigation entry
    nav[parts] = doc_path.as_posix()

    # Write the stub page with the mkdocstrings directive
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}\n")

    # Set the edit path so "edit on GitHub" links point to the source
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(src))

# Write the navigation summary consumed by literate-nav
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
