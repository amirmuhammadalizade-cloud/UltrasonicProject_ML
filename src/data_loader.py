import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_files(
    data_path: Path, x_path: Path, y_path: Path, z_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """چهار فایل npy مربوط به یک نمونه اندازه‌گیری التراسونیک را بارگذاری می‌کند.

    Args:
        data_path: مسیر آرایه سه‌بعدی داده خام (x, y, time).
        x_path: مسیر مختصات محور X.
        y_path: مسیر مختصات محور Y.
        z_path: مسیر محور زمان.

    Raises:
        FileNotFoundError: اگر هر یک از فایل‌ها موجود نباشد.
        ValueError: اگر ابعاد داده خام با طول x یا y همخوانی نداشته باشد.
    """
    for path in (data_path, x_path, y_path, z_path):
        if not path.exists():
            raise FileNotFoundError(f"فایل داده پیدا نشد: {path}")

    raw_data = np.load(data_path, mmap_mode="r")
    x = np.load(x_path)
    y = np.load(y_path)
    z = np.load(z_path)

    if raw_data.shape[0] != len(x) or raw_data.shape[1] != len(y):
        raise ValueError(
            f"ابعاد داده خام {raw_data.shape[:2]} با طول x ({len(x)}) "
            f"یا y ({len(y)}) همخوانی ندارد."
        )

    return raw_data, x, y, z


def load_sample(sample_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """یک نمونه اندازه‌گیری را بر اساس نام فایل داده اصلی بارگذاری می‌کند."""
    return load_files(
        data_path=DATA_DIR / f"{sample_name}.npy",
        x_path=DATA_DIR / "X-values.npy",
        y_path=DATA_DIR / "Y-values.npy",
        z_path=DATA_DIR / "Z-values.npy",
    )