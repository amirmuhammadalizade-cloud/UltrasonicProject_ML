# Ultrasonic Concrete Thickness Estimation

A research-oriented prototype for **ultrasonic pulse-echo signal analysis**, **backwall echo detection**, and **concrete thickness estimation** using signal processing and machine learning.

## Project Goal

The project investigates whether ultrasonic A-scan signals can be processed to estimate concrete thickness reliably.

```text
Raw A-scan
    ↓
Data validation
    ↓
Signal processing
    ↓
Envelope extraction
    ↓
Backwall echo detection
    ↓
Feature engineering
    ↓
Regression / Machine Learning
    ↓
Estimated thickness
    ↓
Spatial visualization
```

The project follows a physics-informed workflow rather than immediately applying complex deep-learning models.

## Project Structure

The codebase has moved from exploratory notebook cells into a small reusable `src/` package. Notebooks are now used for exploration and reporting only; the actual pipeline logic lives in modules.

```text
.
├── notebooks/
│   └── UT_ML.ipynb          # exploratory analysis, plots, physical validation
├── src/
│   ├── data_loader.py        # loads the four .npy files for a given sample
│   ├── preprocessing.py      # valid-region mask, crop to measured grid
│   ├── signal_processing.py  # Signal, Filtering, feature extractors, EchoDetector, DepthEstimator
│   ├── depth_estimation.py   # DepthCalibration, GroundTruthDepth, DatasetBuilder
│   └── dataset.xlsx          # generated feature dataset (output of DatasetBuilder)
└── README.md
```

Data files (`*.npy`) are intentionally excluded from version control (see `.gitignore`) and are expected under `src/../data/`.

## Dataset

The primary dataset is **Pk050**, acquired at the **Bundesanstalt für Materialforschung und -prüfung (BAM), Berlin**, using the pulse-echo ultrasonic method.

### Specimen

- Material: heterogeneous concrete
- Length: 2000 mm
- Width: 800 mm
- Maximum aggregate size: 16 mm
- Step-wise thicknesses:
  - 571.3 mm
  - 452.0 mm
  - 330.9 mm
  - 210.8 mm

### Measurement

- Complete grid: 201 × 81
- Actual measurement grid: 187 × 71
- Spacing: 10 mm
- Samples per A-scan: 4000
- Sampling rate: 2 MS/s
- Sampling interval: 0.5 µs

Main NumPy dataset:

```text
Pk050_3D_Dataset_Long_Rot00.npy
Shape: (201, 81, 4000)
```

## Work Completed

- [x] Dataset documentation and structure analysis
- [x] Zero-vector investigation
- [x] Valid measurement-region extraction
- [x] Spatial/RMS analysis
- [x] Representative A-scan inspection
- [x] Raw signal analysis
- [x] Envelope extraction using the Hilbert transform
- [x] Time-gate based echo investigation
- [x] Preliminary Backwall Echo detection
- [x] Physical validation using four known thicknesses
- [x] Thickness vs. Backwall TOF regression
- [x] Refactor exploratory code into a reusable `src/` package
- [x] Object-oriented feature extraction pipeline (time-domain + frequency-domain)
- [x] Full-grid dataset generation (`DatasetBuilder` → `dataset.xlsx`)
- [ ] Frequency-domain feature validation (see *Known Issues*)
- [ ] Classical ML baseline
- [ ] Spatially-aware validation

## Zero-Vector Investigation

The dataset contains zero-filled positions representing missing measurement points.

Observed:

```text
Total A-scans: 16281
Zero A-scans:   2069
Percentage:     12.71%
```

These points were separated from the valid measurement region before signal analysis (`preprocessing.crop_to_measured_region`, `Signal.non_empty_value_mask`).

## Signal Processing

The analytic signal is used to calculate the envelope:

```python
from scipy.signal import hilbert

analytic_signal = hilbert(a_scan)
envelope = np.abs(analytic_signal)
```

The envelope makes wave-packet and echo localization easier than direct inspection of the oscillating raw A-scan.

A key observation is that the strongest peak in a raw A-scan is **not automatically the Backwall Echo**. Early probe/surface reflections and other internal reflections can be stronger.

This logic is now encapsulated in `signal_processing.py`:

- **`Signal`** — raw grid access, envelope computation, valid-point mask, real time axis (from `Z-values.npy`).
- **`Filtering`** — a configurable, ordered chain of filters (`gaussian`, `median`, `uniform`, `savgol`) applied to the envelope.
- **`TimeDomainFeatures`** — peak amplitude, time of flight, rise time, signal energy, RMS.
- **`FrequencyDomainFeatures`** — dominant frequency, spectral centroid, bandwidth (via FFT).
- **`EchoDetector`** — multi-echo peak detection on an A-scan.
- **`DepthEstimator`** — physics-based `d = v·t/2` conversion, kept as an independent sanity check alongside the data-driven calibration.

## Backwall Echo Validation

Four representative positions were analyzed:

| X (mm) | Y (mm) | Thickness (mm) | Detected TOF (µs) |
|---:|---:|---:|---:|
| 250 | 400 | 571.3 | 273.0 |
| 750 | 400 | 452.0 | 230.5 |
| 1250 | 400 | 330.9 | 170.0 |
| 1750 | 400 | 210.8 | 113.0 |

The preliminary linear fit produced:

```text
slope     = 0.4495 µs/mm
intercept = 20.7498 µs
R²        = 0.9952
```

This is a promising physical sanity check, but it is **not sufficient to claim general model validity**, because only four reference points were used.

Using the simplified pulse-echo relation

```text
t ≈ 2d / v
```

the fitted slope corresponds to an approximate propagation velocity of 4.45 km/s.

This calibration is now formalized as `DepthCalibration` in `depth_estimation.py`, fit once from the four known points and reused as an **input feature** (`calibrated_depth_estimate`), never as the ground-truth label — the true label always comes from the specimen's known step geometry (`GroundTruthDepth`), to avoid circular reasoning in the ML model.

## Feature Engineering — Current Stage

The project has moved from a candidate feature list into a working, full-grid feature extraction pipeline.

`DatasetBuilder` runs the complete pipeline over every valid measurement point and writes one row per point to `src/dataset.xlsx`.

**Current dataset:** 13,277 rows × 15 columns.

| Column | Description |
|---|---|
| `x_index`, `y_index` | grid indices |
| `x_position_mm`, `y_position_mm` | physical position |
| `peak_amplitude` | envelope peak amplitude |
| `time_of_flight` | time of the envelope peak (µs) |
| `rise_time` | 10%–90% rise time before the peak |
| `signal_energy` | Σ signal² |
| `rms` | root-mean-square amplitude |
| `dominant_frequency` | FFT peak frequency |
| `spectral_centroid` | FFT-weighted mean frequency |
| `bandwidth` | −6 dB bandwidth around the FFT peak |
| `snr` | signal-to-noise ratio (dB) vs. an early noise window |
| `calibrated_depth_estimate` | depth from `DepthCalibration` (feature, not label) |
| `true_depth` | depth from specimen geometry (label) |

### Known Issues

- **`dominant_frequency` / `bandwidth` are currently unreliable.** Both are computed by `FrequencyDomainFeatures` on the **Hilbert envelope** rather than the raw RF A-scan. Since the envelope is non-negative, its FFT is dominated by the DC component, which pins `dominant_frequency` to 0 Hz for essentially every point and collapses `bandwidth` to near-zero. Fix in progress: compute frequency-domain features from the raw (pre-envelope) A-scan instead, while keeping envelope-based time-domain features as they are.

## Machine Learning Plan

This is fundamentally a regression problem:

```text
Ultrasonic signal/features → Concrete thickness
```

The planned progression is:

1. Physics-based TOF → thickness
2. Linear Regression
3. Random Forest
4. Gradient Boosting
5. Other suitable classical regressors
6. Deep learning only if justified

Evaluation will include:

- MAE
- RMSE
- R²
- Error distribution
- Robustness
- Physical plausibility

## Important: Spatial Data Leakage

Because the specimen is step-shaped, spatial position is strongly correlated with thickness.

A model that uses `X` may learn:

```text
X position → thickness
```

instead of:

```text
Ultrasonic signal → thickness
```

Random train/test splitting can also leak information because nearby measurement points are highly related.

Therefore, later validation should include spatially aware experiments:

- Signal-only features
- Signal + X/Y
- Held-out spatial regions
- Testing on spatial locations not represented in training

## Prototype Vision

The eventual prototype should be able to:

```text
Select measurement point
        ↓
Display A-scan
        ↓
Display envelope
        ↓
Detect Backwall Echo
        ↓
Calculate TOF
        ↓
Extract features
        ↓
Predict thickness
        ↓
Display prediction error
        ↓
Update thickness map
```

The long-term objective is to develop the foundation for an **Ultrasonic NDT / Structural Health Monitoring** analysis tool.

## Long-Term Direction: UltraPy

The broader project concept may eventually evolve into an engineering platform containing:

- Physics Engine
- Instrument Simulation
- Signal Processing
- Imaging Engine
- AI/ML Engine
- Dataset Factory
- Visualization
- API
- Plugin SDK
- Desktop Application

These are long-term goals. The immediate priority is a scientifically defensible first prototype.

## Research Philosophy

The project follows:

```text
Understand the measurement
        ↓
Understand the signal
        ↓
Understand the physics
        ↓
Build signal-processing methods
        ↓
Validate against known information
        ↓
Extract features
        ↓
Build baseline ML models
        ↓
Test spatial generalization
        ↓
Increase model complexity only if justified
```

The objective is not simply to maximize an ML metric. The objective is to determine whether ultrasonic signal characteristics can provide reliable information about concrete thickness.

## Reference

Maack, S., Küttenbaum, S., Bühling, B., Borchardt-Giers, K., Aßmann, N., Niederleithinger, E. (2023).

**Low frequency ultrasonic pulse-echo datasets for object detection and thickness measurement in concrete specimens as testing tasks in civil engineering.**

Data in Brief, 48, 109233.

DOI: `10.1016/j.dib.2023.109233`

## Development Status

```text
Dataset Understanding       ████████████████████ 100%
Signal Analysis              ████████████████████ 100%
Echo Investigation           ████████████████████ 100%
Physical Validation          ████████████████████ 100%
Feature Engineering          ████████████████░░░░  80%
Classical ML                 ░░░░░░░░░░░░░░░░░░░░   0%
Spatial Validation           ░░░░░░░░░░░░░░░░░░░░   0%
Deep Learning                ░░░░░░░░░░░░░░░░░░░░   0%
Prototype UI                 ░░░░░░░░░░░░░░░░░░░░   0%
```

**Current milestone: Fix frequency-domain features → Classical ML → Spatial Validation**
