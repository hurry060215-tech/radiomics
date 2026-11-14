Build: python -m sphinx -W --keep-going -b html website website/_build/html
Preview: python -m http.server 8765 --directory website/_build/html
No deployment is configured. Tutorial execution is separate from HTML rendering.
