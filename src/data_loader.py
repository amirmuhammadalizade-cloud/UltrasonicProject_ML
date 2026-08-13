import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def load_files(data: Path, x: Path, y: Path, z: Path):
    raw_data = np.load(data, mmap_mode="r")
    X = np.load(x)
    Y = np.load(y)
    Z = np.load(z)
    return raw_data, X, Y, Z


def loading():
    raw_data , x, y, z = load_files(
        data = DATA_DIR / "Pk050_3D_Dataset_Long_Rot00.npy",
        x = DATA_DIR / "X-values.npy",
        y = DATA_DIR / "Y-values.npy",
        z =  DATA_DIR / "Z-values.npy"
    )
    return raw_data, x, y, z

