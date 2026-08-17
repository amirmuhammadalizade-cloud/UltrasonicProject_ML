from loading import File
import numpy as np



# ====================
# داده های صفر رو جدا می کنیم
# ====================
file = File()

def is_zero_signal(signal):
    """

    """
    return np.all(signal != 0, axis=2)

mask = is_zero_signal(file.data)
print(len(mask))