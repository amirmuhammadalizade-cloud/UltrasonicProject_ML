import numpy as np
import pandas as pd

from data_loader import load_sample
from signal_processing import (
    Signal,
    apply_filters,
    extract_features,
    groundtruth_depth,
    restrict_to_window,
    snr,
)


def build_dataset(signal_obj, filter_steps, output_path, t_min=80.0, t_max=320.0):
    time = signal_obj.time()
    time_step_us = float(time[1] - time[0])
    sampling_rate_hz = 1.0 / (time_step_us * 1e-6)

    x_indices, y_indices = np.nonzero(signal_obj.position_mask)

    rows = []
    for x_index, y_index in zip(x_indices, y_indices):
        envelope = signal_obj.compute_envelope(x_index, y_index)
        filtered_envelope = apply_filters(envelope, filter_steps)
        raw_signal = signal_obj.raw_signal(x_index, y_index)

        # SNR must be computed on the FULL (unwindowed) envelope, since the
        # windowed segment no longer contains a pure-noise region to compare
        # the peak against.
        snr_value = snr(filtered_envelope)

        windowed_envelope, windowed_time = restrict_to_window(
            filtered_envelope, time, t_min, t_max
        )
        windowed_raw, _ = restrict_to_window(raw_signal, time, t_min, t_max)
        features = extract_features(
            windowed_envelope, windowed_raw, windowed_time, sampling_rate_hz
        )

        row = {
            "x_position_mm": float(signal_obj.x[x_index]),
            "y_position_mm": float(signal_obj.y[y_index]),
            "groundtruth": groundtruth_depth(float(signal_obj.x[x_index])),
            "snr": snr_value,
            **features.to_dict(),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    raw_data, x_axis_mm, y_axis_mm, z_axis = load_sample("Pk050_3D_Dataset_Long_Rot00")
    signal_obj = Signal(data=raw_data, x=x_axis_mm, y=y_axis_mm, z=z_axis)

    df = build_dataset(
        signal_obj=signal_obj,
        filter_steps=[("gaussian", {"sigma": 15})],
        output_path="dataset.csv",
    )

    # sanity check: groundtruth vs time_of_flight should now correlate strongly
    print(df[["x_position_mm", "groundtruth", "time_of_flight"]].corr())
    print(df.groupby("groundtruth")["time_of_flight"].agg(["mean", "std"]))