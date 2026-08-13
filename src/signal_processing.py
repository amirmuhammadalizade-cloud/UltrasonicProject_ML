from matplotlib import pyplot as plt

from data_loader import loading
import numpy as np


class Signal:

    def __init__(self, data, x, y, z):
        self.data = data
        self.x = x
        self.y = y
        self.z = z


    def show(self, x_index, y_index):
        import matplotlib.pyplot as plt

        a_scan = self.data[x_index, y_index, :]

        plt.figure(figsize=(12, 5))
        plt.plot(
            self.z,
            a_scan,
            label=f"A-scan X index:{x_index}, Y index:{y_index}"
        )

        plt.xlabel("Time (µs)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()


    def non_empty_value_mask(self):
        mask = np.any(self.data != 0, axis=2)
        return mask


    def compute_envelope(self, x_index, y_index):
        from scipy.signal import hilbert

        mask = self.non_empty_value_mask()

        if not mask[x_index, y_index]:
            raise ValueError("Selected measurement point is empty.")

        a_scan = self.data[x_index, y_index, :]

        analytic_signal = hilbert(a_scan)
        envelope = np.abs(analytic_signal)

        return envelope


    def Time(self):
        time = np.arange(len(self.z)) * 0.5
        return time


data, x, y, z = loading()
k050 = Signal(data, x, y, z)

plt.plot(k050.Time(), k050.compute_envelope(50, 10))
plt.show()

