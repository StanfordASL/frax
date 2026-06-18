import importlib.resources as resources
from pathlib import Path

ASSETS_DIR = Path(resources.files("frax.assets"))

FRANKA_ASSETS_DIR = ASSETS_DIR / "franka_panda"
G1_ASSETS_DIR = ASSETS_DIR / "unitree_g1"
KUKA_ASSETS_DIR = ASSETS_DIR / "iiwa"
