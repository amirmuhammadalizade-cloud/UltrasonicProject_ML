import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_sample
from signal_processing import (
    Signal,
    apply_filters,
    restrict_to_window,
    strongest_echo,
    find_local_maxima_in_window,
    snr,
    groundtruth_depth,
)

raw_data, x_axis_mm, y_axis_mm, z_axis = load_sample("Pk050_3D_Dataset_Long_Rot00")
signal_obj = Signal(data=raw_data, x=x_axis_mm, y=y_axis_mm, z=z_axis)

time = signal_obj.time()
t_min, t_max = 50, 320.0

x_indices, y_indices = np.nonzero(signal_obj.position_mask)

target_indices = [
    (x_index, y_index)
    for x_index, y_index in zip(x_indices, y_indices)
    if groundtruth_depth(float(signal_obj.x[x_index])) == 571.3
]

n_samples = 6
step = max(1, len(target_indices) // n_samples)
sample_points = target_indices[::step][:n_samples]

fig, axes = plt.subplots(len(sample_points), 1, figsize=(10, 3 * len(sample_points)))

for ax, (x_index, y_index) in zip(axes, sample_points):
    envelope = signal_obj.compute_envelope(x_index, y_index)
    filtered_envelope = apply_filters(envelope, [("gaussian", {"sigma": 20})])

    snr_value = snr(filtered_envelope)
    windowed_envelope, windowed_time = restrict_to_window(filtered_envelope, time, t_min, t_max)

    _, tof_strongest = strongest_echo(windowed_envelope, windowed_time)
    candidates = find_local_maxima_in_window(windowed_envelope, windowed_time, snr_value)

    ax.plot(time, filtered_envelope, label="filtered envelope (full)")
    ax.axvspan(t_min, t_max, color="orange", alpha=0.15, label="search window")
    ax.axvline(tof_strongest, color="red", linestyle="--", label=f"strongest_echo: {tof_strongest:.1f} µs")
    ax.axvline(273.0, color="green", linestyle=":", label="expected ~273 µs")

    if candidates:
        cand_times = [c[0] for c in candidates]
        cand_amps = [c[1] for c in candidates]
        ax.scatter(cand_times, cand_amps, color="purple", zorder=5, s=60,
                   label=f"candidates ({len(candidates)})")

    x_pos = float(signal_obj.x[x_index])
    y_pos = float(signal_obj.y[y_index])
    ax.set_title(f"x={x_pos:.0f}mm, y={y_pos:.0f}mm — snr={snr_value:.2f}")
    ax.legend(fontsize=8)
    ax.set_xlabel("Time (µs)")

plt.tight_layout()
plt.savefig("local_maxima_candidates_debug.png", dpi=120)
plt.show()