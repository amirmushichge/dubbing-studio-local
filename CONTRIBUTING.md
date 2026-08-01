# Contributing

1. Create a fork and a focused branch.
2. Install the lightweight environment: `py -3.10 -m venv .venv`, then `.\.venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt`.
3. For UI work without models, set `$env:DUBBING_STUDIO_MOCK='1'` and run `start.ps1`.
4. Before opening a pull request, run `pytest -q`, `python -m compileall -q app workers tools tests` and `node --check static/app.js`.

Do not commit model weights, virtual environments, user videos, exports or logs. Changes to voice cloning must preserve the explicit consent requirement. UI changes must keep English product copy and the documented AmirStyle typography, grid and accessibility rules.
