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

## Zero-Vector Investigation

The dataset contains zero-filled positions representing missing measurement points.

Observed:

```text
Total A-scans: 16281
Zero A-scans:   2069
Percentage:     12.71%
```

These points were separated from the valid measurement region before signal analysis.

## Signal Processing

The analytic signal is used to calculate the envelope:

```python
from scipy.signal import hilbert

analytic_signal = hilbert(a_scan)
envelope = np.abs(analytic_signal)
```

The envelope makes wave-packet and echo localization easier than direct inspection of the oscillating raw A-scan.

A key observation is that the strongest peak in a raw A-scan is **not automatically the Backwall Echo**. Early probe/surface reflections and other internal reflections can be stronger.

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

## Current Stage

The project is now entering **Feature Engineering**.

The first goal is to create an interpretable feature extractor for individual A-scans and validate it on the four known reference points before processing the entire dataset.

### Initial candidate features

Time domain:

- Mean
- Standard deviation
- RMS
- Maximum amplitude
- Minimum amplitude
- Peak-to-peak amplitude
- Signal energy

Envelope:

- Maximum envelope
- Envelope energy
- Number of peaks
- Peak positions
- Peak amplitudes

Backwall:

- Backwall TOF
- Backwall envelope amplitude
- Backwall peak amplitude
- Local energy around the Backwall Echo

Later, frequency-domain features can be added:

- Dominant frequency
- Spectral energy
- Spectral centroid
- Bandwidth

A first feature table may look like:

```text
X
Y
RMS
STD
Energy
Envelope_Max
BW_TOF
BW_Amplitude
Thickness
```

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
Signal Analysis             ████████████████████ 100%
Echo Investigation          ████████████████████ 100%
Physical Validation         ████████████████████ 100%
Feature Engineering         ░░░░░░░░░░░░░░░░░░░░   0%
Classical ML                ░░░░░░░░░░░░░░░░░░░░   0%
Spatial Validation          ░░░░░░░░░░░░░░░░░░░░   0%
Deep Learning               ░░░░░░░░░░░░░░░░░░░░   0%
Prototype UI                ░░░░░░░░░░░░░░░░░░░░   0%
```

**Current milestone: Feature Engineering → Classical ML → Spatial Validation**
