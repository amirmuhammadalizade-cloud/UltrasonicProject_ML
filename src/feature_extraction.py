from signal_processing import Signal
from data_loader import loading
from matplotlib import pyplot as plt

data, x, y, z = loading()
k050 = Signal(data, x, y, z)

plt.plot(k050.Time(), k050.compute_envelope(50, 10))
plt.show()