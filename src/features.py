from loading import File

import numpy as np

from scipy.signal import (
    argrelextrema,
    find_peaks,
    butter,
    sosfiltfilt
    )

from scipy.fft import rfft, irfft, next_fast_len
import matplotlib as plt


# ========================================
# داده های صفر رو جدا می کنیم
# ========================================

file = File()

def not_zero_signal(signal):
    """
    این تابع بررسی می کنه کدام نقاط صفر نیستند
    یک ماسک بولین به ما خروجی می ده
    """
    return np.all(signal != 0, axis=2)


# ========================================
# فیلتر فرکانسی
# ========================================

def butterworth_bandpass(
    signal,
    fs,
    lowcut,
    highcut,
    order=4,
    padtype="odd"
):
    """
    اعمال فیلتر Butterworth Band-pass روی سیگنال.

    Parameters
    ----------
    signal : array-like
        سیگنال ورودی یک‌بعدی.

    fs : float
        نرخ نمونه‌برداری بر حسب Hz.

    lowcut : float
        فرکانس قطع پایین بر حسب Hz.

    highcut : float
        فرکانس قطع بالا بر حسب Hz.

    order : int, optional
        مرتبه فیلتر. مقدار پیش‌فرض 5 است.

    padtype : str or None, optional
        روش padding برای sosfiltfilt.

    Returns
    -------
    filtered_signal : numpy.ndarray
        سیگنال فیلترشده.
    """

    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise ValueError(
            "signal باید یک آرایه‌ی یک‌بعدی باشد."
        )

    if len(signal) < 10:
        raise ValueError(
            "طول سیگنال برای فیلتر کردن کافی نیست."
        )

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(
            "fs باید یک عدد مثبت و معتبر باشد."
        )

    if not isinstance(order, int) or order < 1:
        raise ValueError(
            "order باید یک عدد صحیح بزرگ‌تر از صفر باشد."
        )

    if not np.isfinite(lowcut) or not np.isfinite(highcut):
        raise ValueError(
            "فرکانس‌های قطع باید مقدار معتبر داشته باشند."
        )

    nyquist = fs / 2

    if lowcut <= 0:
        raise ValueError(
            "lowcut باید بزرگ‌تر از صفر باشد."
        )

    if highcut >= nyquist:
        raise ValueError(
            "highcut باید کمتر از فرکانس Nyquist باشد."
        )

    if lowcut >= highcut:
        raise ValueError(
            "lowcut باید کمتر از highcut باشد."
        )

    sos = butter(
        N=order,
        Wn=[lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    filtered_signal = sosfiltfilt(
        sos,
        signal,
        padtype=padtype
    )

    return filtered_signal

# ========================================
# تکرار شوندگی تابع رو پیدا می کنیم 
# ========================================

def echo_periodicity_autocorr(
    signal,
    time,
    min_period=80,
    max_period=350,
    min_prominence=0.05,
    min_peak_distance=None,
    analysis_start_time=120,
    analysis_end_time=None,
):
    """
    Estimate echo periodicity using FFT-based autocorrelation.

    Parameters
    ----------
    signal : array-like
        Input ultrasonic signal.
    time : array-like
        Time axis.
    min_period : float
        Minimum expected echo period.
    max_period : float
        Maximum expected echo period.
    min_prominence : float
        Minimum peak prominence in normalized autocorrelation.
    min_peak_distance : float or None
        Minimum distance between peaks in time units.

    Returns
    -------
    period : float
        Estimated echo period.
    strength : float
        Autocorrelation value at the selected lag.
    """

    signal = np.asarray(signal, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)

    # Validation

    if signal.ndim != 1 or time.ndim != 1:
        raise ValueError("signal و time باید آرایه‌ی یک‌بعدی باشند.")

    if signal.size != time.size:
        raise ValueError("signal و time باید طول یکسان داشته باشند.")

    if signal.size < 4:
        raise ValueError("سیگنال خیلی کوتاه است.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("signal شامل مقادیر نامعتبر است.")

    if not np.all(np.isfinite(time)):
        raise ValueError("time شامل مقادیر نامعتبر است.")

    # Restrict analysis region

    mask = time >= analysis_start_time

    if analysis_end_time is not None:
        mask &= time <= analysis_end_time

    if np.count_nonzero(mask) < 4:
        raise ValueError("بازه‌ی تحلیل برای محاسبه‌ی autocorrelation خیلی کوتاه است.")

    signal = signal[mask]
    time = time[mask]

    # Sampling interval

    dt_values = np.diff(time)
    dt = np.mean(dt_values)

    if dt <= 0: 
        raise ValueError("dt نامعتبر است.")

    if not np.allclose(dt_values, dt, rtol=1e-4, atol=1e-12):
        raise ValueError("time باید تقریباً یکنواخت باشد.")

    # Remove DC

    x = signal - np.mean(signal)

    energy = np.dot(x, x)

    if energy <= np.finfo(float).eps:
        raise ValueError("انرژی سیگنال تقریباً صفر است.")

    # FFT autocorrelation

    n = len(x)
    n_fft = next_fast_len(2 * n - 1)
    spectrum = rfft(x, n=n_fft)
    autocorr = irfft(
        spectrum * np.conjugate(spectrum), # type: ignore
        n=n_fft)[:n]

    # Normalize
    autocorr /= autocorr[0] # type: ignore

    # Convert period range -> lag range

    lag_min = max(1,
        int(round(min_period / dt)))

    lag_max = min(
        int(round(max_period / dt)),
        len(autocorr) - 1)

    if lag_max <= lag_min:
        raise ValueError("بازه‌ی period با طول سیگنال سازگار نیست.")

    search_region = autocorr[lag_min:lag_max + 1]

    # Peak detection

    if min_peak_distance is None:
        min_peak_distance = min_period * 0.5

    distance_samples = max(1,
        int(round(min_peak_distance / dt)))

    peaks, properties = find_peaks(
        search_region,
        prominence=min_prominence,
        distance=distance_samples)

    # Fallback

    if len(peaks) == 0:
        best_rel = int(np.argmax(search_region))

    else:
        prominences = properties["prominences"]
        heights = search_region[peaks]

        # Normalize
        height_score = (
            heights / np.max(heights))

        prominence_score = (
            prominences / np.max(prominences))

        # Combined score
        scores = (
            0.5 * height_score +
            0.5 * prominence_score)

        best_index = np.argmax(scores)
        best_rel = int(peaks[best_index])

    # Final result

    best_lag = lag_min + best_rel
    period = best_lag * dt
    strength = autocorr[best_lag]

    if not np.isfinite(period):
        raise ValueError("period نامعتبر است.")

    return period, strength


# ========================================
# تست کردن
# ========================================


a_scan = file.data[175, 6, :]

time = file.t

period, strength = echo_periodicity_autocorr(
    signal=a_scan,
    time=time,
    min_period=80,
    max_period=350,
    min_prominence=0.05,
    min_peak_distance=None,
    analysis_start_time=350,
    analysis_end_time=None,
    )
print(f'period: {period}', f'\nstrength: {strength}')
