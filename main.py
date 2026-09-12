from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt


class Data:
    def __init__(self):
        self.scan = None
        self.x = None
        self.y = None
        self.z = None

    def import_file(self):
        BASE_DIR = Path(__file__).resolve().parent
        file_path = str(BASE_DIR / "data")

        scan_path = file_path + "/Pk050_3D_Dataset_Long_Rot00.npy"
        self.scan = np.load(scan_path)

        x_path = file_path + "/X-values.npy"
        self.x = np.load(x_path)

        y_path = file_path + "/Y-values.npy"
        self.y = np.load(y_path)

        z_path = file_path + "/Z-values.npy"
        self.z = np.load(z_path)



data = Data()
data.import_file()

class Signal:
    def __init__(self, data):
        self.data = data

    def a_scan(self, x_position, y_position):
        t = np.arange(0, len(self.data)-1, 0.05)
        return self.data[x_position, y_position, :] , t



signal = Signal(data.scan)

s, t = signal.a_scan(25, 11)


plt.plot(t, s)
plt.show()