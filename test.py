import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\Users\asus\PycharmProjects\UltrasonicProject_ML\data")
data = np.load(DATA_DIR / "Pk050_3D_Dataset_Long_Rot00.npy", mmap_mode="r")
x_coords = np.load(DATA_DIR / "X-values.npy")
y_coords = np.load(DATA_DIR / "Y-values.npy")

print("برای هر X، چند درصد از Y ها سیگنال واقعی (غیرصفر) دارن:")
for x_idx in range(0, data.shape[0], 5):  # هر ۵ تا یکی، برای خلاصه بودن
    signals = data[x_idx, :, :]
    nonzero_fraction = np.mean(np.any(signals != 0, axis=1))
    print(f"  X={x_coords[x_idx]:6.0f} mm  ->  {nonzero_fraction*100:5.1f}% از Y ها سیگنال دارن")