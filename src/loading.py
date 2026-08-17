"""
بارگذاری داده 
داخل این بخش داده های آلتروسونیک را وارد پروژه می کنیم
"""

from pathlib import Path
import numpy as np

class File:
    def __init__(self) -> None:
        pass
        DATA_DIR = Path(r"C:\Users\asus\PycharmProjects\UltrasonicProject_ML\data")
        self.data = np.load(
            DATA_DIR / "Pk050_3D_Dataset_Long_Rot00.npy",
            mmap_mode="r"
        )

        self.x = np.load(DATA_DIR / "X-values.npy")
        self.y = np.load(DATA_DIR / "Y-values.npy")
        self.t = np.load(DATA_DIR / "Z-values.npy")

file = File()
print(file.t)