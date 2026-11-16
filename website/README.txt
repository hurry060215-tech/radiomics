Prepare: python tools/build_website_sources.py
Build: python -m sphinx -W --keep-going -b html website website/_build/html
Preview: python -m http.server 8765 --directory website/_build/html
GitHub Pages: https://hurry060215-tech.github.io/radiomics/
PR: preview only. Main pushes or explicit maintainer workflow dispatches publish allowed branches.
Tutorial execution is separate from HTML rendering.
