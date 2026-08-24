# Contributing

1. Create a fork and a focused branch.
2. Create a Python 3.10 virtual environment and install `requirements.txt` plus `requirements-dev.txt`.
3. For UI work without models, set `DUBBING_STUDIO_MOCK=1` and run the platform launcher.
4. Before opening a pull request, run the compile, Ruff, pytest and Node commands from the pull-request checklist.
5. Parse Windows launchers with PowerShell and macOS launchers with `bash -n setup.command start.command doctor.command` when changing setup or startup behavior.

Do not commit model weights, virtual environments, user videos, exports, logs, local paths or credentials. Use synthetic media for public screenshots and test fixtures. Changes to voice cloning must preserve the explicit consent requirement. UI changes must keep English product copy and the documented AmirStyle typography, grid and accessibility rules.
