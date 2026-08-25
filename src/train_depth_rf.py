import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, r2_score
import joblib

from loading import File
from features import (
    not_zero_signal,
    butterworth_bandpass,
    echo_periodicity_autocorr,
    local_power_ratio,
    local_std_ratio,
    local_skewness_shift,
    local_crest_factor_ratio,
    local_spectral_centroid_ratio,
    local_dominant_frequency_ratio,
    local_entropy_ratio,
)

# ================= تنظیمات =================
SIGNAL_FILES = [
    "Pk050_3D_Dataset_Long_Rot00.npy",
    # فایل‌های دیگه (Shear_Rot90, Shear_Rot00, ...) رو هم اینجا اضافه کن
]

FS = 2_000_000
LOWCUT, HIGHCUT = 20_000, 90_000  # Hz - با فرکانس مرکزی پراب تنظیم شود
ACCURACY = 10

STEP_LENGTH = 500.0
DEPTHS = [571.3, 452.0, 330.9, 210.8]  # mm - d1..d4

N_ESTIMATORS = 300
TEST_SIZE = 0.2
RANDOM_STATE = 42

RATIO_FUNCS_1D = [local_power_ratio, local_std_ratio, local_crest_factor_ratio, local_skewness_shift]
RATIO_FUNCS_FS = [local_spectral_centroid_ratio, local_dominant_frequency_ratio]


def depth_from_x(x_mm):
    idx = min(int(x_mm // STEP_LENGTH), len(DEPTHS) - 1)
    return DEPTHS[idx]


def summarize(arr):
    arr = arr[np.isfinite(arr)]
    return [arr.mean(), arr.std(), arr.max()] if arr.size else [0.0, 0.0, 0.0]


def extract_features(signal, time):
    sig = butterworth_bandpass(signal, FS, LOWCUT, HIGHCUT)
    period, strength = echo_periodicity_autocorr(sig, time)
    feats = [period, strength]

    for fn in RATIO_FUNCS_1D:
        feats += summarize(fn(sig, ACCURACY))
    for fn in RATIO_FUNCS_FS:
        feats += summarize(fn(sig, FS, ACCURACY))
    feats += summarize(local_entropy_ratio(sig, ACCURACY))

    return feats


def build_dataset():
    X_rows, y_rows = [], []

    for fname in SIGNAL_FILES:
        f = File(fname)
        signals, x_coords, time = f.data, f.x, f.t
        valid = not_zero_signal(signals)  # (n_x, n_y)

        for i in range(signals.shape[0]):
            for j in range(signals.shape[1]):
                if not valid[i, j]:
                    continue
                sig = np.asarray(signals[i, j], dtype=float)
                X_rows.append(extract_features(sig, time))
                y_rows.append(depth_from_x(x_coords[i]))

    return np.array(X_rows), np.array(y_rows)


def main():
    X, y = build_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"MAE (mm): {mean_absolute_error(y_test, pred):.2f}")
    print(f"R2: {r2_score(y_test, pred):.4f}")

    for d in DEPTHS:
        mask = y_test == d
        if mask.any():
            mae_d = mean_absolute_error(y_test[mask], pred[mask])
            print(f"  depth {d} mm -> MAE {mae_d:.2f} mm  (n={mask.sum()})")

    joblib.dump(model, "depth_rf_model.joblib")


if __name__ == "__main__":
    main()