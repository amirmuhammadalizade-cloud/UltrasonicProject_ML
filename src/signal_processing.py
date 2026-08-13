import numpy as np
from scipy.ndimage import (
    gaussian_filter1d,
    median_filter,
    uniform_filter1d,
)
from scipy.signal import find_peaks, savgol_filter, hilbert


class Signal:

    def __init__(self, data, x, y, z):
        self.data = data
        self.x = x
        self.y = y
        self.z = z

    def non_empty_value_mask(self):
        mask = np.any(self.data != 0, axis=2)
        return mask

    def compute_envelope(self, x_index, y_index) -> np.ndarray:
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

    def set_filter(
        self,
        signal,
        gaussian_f_s=None,
        median_f_s=None,
        uniform_f_s=None,
        savgol_f_a=(None, None),
    ):

        filtering = Filtering(
            signal=signal,
            gaussian_f_sigma=gaussian_f_s,
            median_f_size=median_f_s,
            uniform_f_size=uniform_f_s,
            savgol_f_amount=savgol_f_a,
        )

        return filtering


class Filtering:

    def __init__(
        self,
        signal: np.ndarray,
        gaussian_f_sigma=None,
        median_f_size=None,
        uniform_f_size=None,
        savgol_f_amount=(None, None),
    ):

        self.signal = signal

        self.gaussian_f_sigma = gaussian_f_sigma
        self.median_f_size = median_f_size
        self.uniform_f_size = uniform_f_size

        self.savgol_f_window = savgol_f_amount[0]
        self.savgol_f_polyorder = savgol_f_amount[1]

    def apply(self):
        signal = self.signal

        if self.gaussian_f_sigma is not None:
            signal = gaussian_filter1d(
                signal,
                self.gaussian_f_sigma
            )

        if self.median_f_size is not None:
            signal = median_filter(
                signal,
                self.median_f_size
            )

        if self.uniform_f_size is not None:
            signal = uniform_filter1d(
                signal,
                self.uniform_f_size
            )

        if self.savgol_f_window is not None:
            signal = savgol_filter(
                signal,
                window_length=self.savgol_f_window,
                polyorder=self.savgol_f_polyorder
            )

        return signal


from data_loader import loading
from matplotlib import pyplot as plt

data, x, y, z = loading()

k050 = Signal(data, x, y, z)

envelope = k050.compute_envelope(50, 10)

filtering = k050.set_filter(
    signal=envelope,
    gaussian_f_s=4,
    median_f_s=5,
    uniform_f_s=22,
    savgol_f_a=(13, 5)
)

filtered_signal = filtering.apply()



plt.plot(k050.Time(), envelope, label= 'envelope', alpha = 0.7)
plt.plot(k050.Time(), filtered_signal ,
          label= 'filtered_signal')

plt.grid(True)
plt.legend()
plt.show()

