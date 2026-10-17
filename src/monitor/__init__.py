"""Local, read-only monitoring and comparison app for training runs (plano E2).

    python -m src.monitor [--runs runs] [--port 8765]

``catalog`` discovers runs and their status, ``readers`` turns a run's artifacts into JSON
payloads, ``compare`` lines several runs up, and ``server`` exposes them to the static page in
``static/``. Nothing here recomputes a result or writes into a run directory.
"""
