from __future__ import annotations

import os
from pathlib import Path

from demucs.pretrained import get_model

torch_home = Path(os.environ["TORCH_HOME"])
torch_home.mkdir(parents=True, exist_ok=True)
print(f"Downloading htdemucs to {torch_home}")
get_model("htdemucs")
print("htdemucs is ready.")
