from __future__ import annotations

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
        raise ValueError("signal باید یک آرایه‌ی یک‌بعدی باشد.")
    if len(signal) < 10:
        raise ValueError("طول سیگنال برای فیلتر کردن کافی نیست.")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs باید یک عدد مثبت و معتبر باشد.")
    if not isinstance(order, int) or order < 1:
        raise ValueError("order باید یک عدد صحیح بزرگ‌تر از صفر باشد.")
    if not np.isfinite(lowcut) or not np.isfinite(highcut):
        raise ValueError("فرکانس‌های قطع باید مقدار معتبر داشته باشند.")

    nyquist = fs / 2

    if lowcut <= 0:
        raise ValueError("lowcut باید بزرگ‌تر از صفر باشد.")
    if highcut >= nyquist:
        raise ValueError("highcut باید کمتر از فرکانس Nyquist باشد.")
    if lowcut >= highcut:
        raise ValueError("lowcut باید کمتر از highcut باشد.")

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

    mask = time >= analysis_start_time

    if analysis_end_time is not None:
        mask &= time <= analysis_end_time

    if np.count_nonzero(mask) < 4:
        raise ValueError("بازه‌ی تحلیل برای محاسبه‌ی autocorrelation خیلی کوتاه است.")

    signal = signal[mask]
    time = time[mask]

    dt_values = np.diff(time)
    dt = np.mean(dt_values)

    if dt <= 0: 
        raise ValueError("dt نامعتبر است.")

    if not np.allclose(dt_values, dt, rtol=1e-4, atol=1e-12):
        raise ValueError("time باید تقریباً یکنواخت باشد.")

    x = signal - np.mean(signal)

    energy = np.dot(x, x)

    if energy <= np.finfo(float).eps:
        raise ValueError("انرژی سیگنال تقریباً صفر است.")

    n = len(x)
    n_fft = next_fast_len(2 * n - 1)
    spectrum = rfft(x, n=n_fft)
    autocorr = irfft(
        spectrum * np.conjugate(spectrum),
        n=n_fft)[:n]

    autocorr /= autocorr[0]

    lag_min = max(1,
        int(round(min_period / dt)))

    lag_max = min(
        int(round(max_period / dt)),
        len(autocorr) - 1)

    if lag_max <= lag_min:
        raise ValueError("بازه‌ی period با طول سیگنال سازگار نیست.")

    search_region = autocorr[lag_min:lag_max + 1]

    if min_peak_distance is None:
        min_peak_distance = min_period * 0.5

    distance_samples = max(1,
        int(round(min_peak_distance / dt)))

    peaks, properties = find_peaks(
        search_region,
        prominence=min_prominence,
        distance=distance_samples)

    if len(peaks) == 0:
        best_rel = int(np.argmax(search_region))

    else:
        prominences = properties["prominences"]
        heights = search_region[peaks]

        height_score = (heights / np.max(heights))

        prominence_score = (prominences / np.max(prominences))

        scores = (
            0.5 * height_score +
            0.5 * prominence_score)

        best_index = np.argmax(scores)
        best_rel = int(peaks[best_index])

    best_lag = lag_min + best_rel
    period = best_lag * dt
    strength = autocorr[best_lag]

    if not np.isfinite(period):
        raise ValueError("period نامعتبر است.")

    return period, strength

# ========================================
# local_power_ratio
# ========================================

def local_power_ratio(signal, accuracy):

    if signal.ndim != 1:
        raise ValueError("سیگنال باید یک بعدی باشد")

    if not np.all(np.isfinite(signal)):
        raise ValueError("signal شامل مقادیر نامعتبر است")

    if not accuracy in range(1,41):
        raise ValueError("دقت باید بین یک تا چهل باشد.")

    power = signal ** 2

    small_size = accuracy

    left = accuracy * 2
    right = accuracy * 2

    n = len(power)

    small_starts = np.arange(0, n, small_size)
    small_ends = np.minimum(small_starts + small_size,n)

    big_starts = np.maximum(0,small_starts - left)
    big_ends = np.minimum(n,small_ends + right)

    cumsum = np.concatenate(([0], np.cumsum(power)))

    small_sums = (cumsum[small_ends]- cumsum[small_starts])
    small_lengths = (small_ends - small_starts)
    small_means = (small_sums / small_lengths)

    big_sums = (cumsum[big_ends] - cumsum[big_starts])
    big_lengths = (big_ends - big_starts)
    big_means = (big_sums / big_lengths)

    ratios = np.divide(
        small_means,
        big_means,
        out=np.zeros_like(small_means),
        where=big_means != 0
    )

    return ratios



# ========================================
# local_std_ratio
# ========================================
def local_std_ratio(signal, accuracy):

    if signal.ndim != 1:
        raise ValueError("سیگنال باید یک بعدی باشد")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal شامل مقادیر نامعتبر است")
    if accuracy not in range(1, 41):
        raise ValueError("دقت باید بین 1 تا 40 باشد")

    small_size = accuracy

    left = accuracy * 2
    right = accuracy * 2

    n = len(signal)

    small_starts = np.arange(0,n,small_size)
    small_ends = np.minimum(small_starts + small_size,n)

    big_starts = np.maximum(0,small_starts - left)
    big_ends = np.minimum(small_ends + right,n)

    cumsum = np.concatenate(([0], np.cumsum(signal)))

    cumsum_sq = np.concatenate(([0], np.cumsum(signal ** 2)))

    small_sums = (cumsum[small_ends] - cumsum[small_starts])

    small_sums_sq = (cumsum_sq[small_ends] - cumsum_sq[small_starts])

    small_lengths = (small_ends - small_starts)

    small_means = (small_sums / small_lengths)

    small_variances = (small_sums_sq / small_lengths - small_means ** 2)

    small_variances = np.maximum(small_variances, 0)

    small_stds = np.sqrt(small_variances)

    big_sums = (cumsum[big_ends] - cumsum[big_starts])

    big_sums_sq = (cumsum_sq[big_ends] - cumsum_sq[big_starts])

    big_lengths = (big_ends - big_starts)

    big_means = (big_sums / big_lengths)

    big_variances = (big_sums_sq / big_lengths - big_means ** 2)

    big_variances = np.maximum(big_variances, 0)

    big_stds = np.sqrt(big_variances)

    ratios = np.divide(
        small_stds,
        big_stds,
        out=np.full_like(
            small_stds,
            np.nan,
            dtype=float
        ),
        where=big_stds != 0
    )

    return ratios






import numpy as np


# ========================================
# local_skewness_shift
# ========================================
def local_skewness_shift(signal, accuracy):
    """
    اختلاف عدم تقارن (skewness) بین پنجره‌ی کوچک و پنجره‌ی بزرگ اطراف آن.

    نکته: برخلاف دو تابع قبلی که خروجی‌شان یک «نسبت» است، اینجا خروجی
    «تفاضل» است، نه نسبت. چون skewness می‌تواند منفی یا نزدیک صفر باشد،
    تقسیم دو مقدار skewness باعث می‌شود علامت خروجی گاهی بی‌معنی شود یا
    وقتی مقدار مخرج نزدیک صفر است، خروجی به‌شدت پرت شود. تفاضل، مشکل
    را ندارد و همان اطلاعات (چقدر شکل پنجره‌ی کوچک از زمینه‌اش نامتقارن‌تر
    شده) را می‌دهد.
    """

    if signal.ndim != 1:
        raise ValueError("سیگنال باید یک بعدی باشد")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal شامل مقادیر نامعتبر است")
    if accuracy not in range(1, 41):
        raise ValueError("دقت باید بین 1 تا 40 باشد")

    small_size = accuracy
    left = accuracy * 2
    right = accuracy * 2

    n = len(signal)

    small_starts = np.arange(0, n, small_size)
    small_ends = np.minimum(small_starts + small_size, n)

    big_starts = np.maximum(0, small_starts - left)
    big_ends = np.minimum(small_ends + right, n)

    cumsum1 = np.concatenate(([0], np.cumsum(signal)))
    cumsum2 = np.concatenate(([0], np.cumsum(signal ** 2)))
    cumsum3 = np.concatenate(([0], np.cumsum(signal ** 3)))

    eps = np.finfo(float).eps

    def _skewness(starts, ends):
        lengths = ends - starts

        sum1 = cumsum1[ends] - cumsum1[starts]
        sum2 = cumsum2[ends] - cumsum2[starts]
        sum3 = cumsum3[ends] - cumsum3[starts]

        mean = sum1 / lengths

        m2 = sum2 / lengths - mean ** 2
        m2 = np.maximum(m2, 0)

        m3 = (
            sum3 / lengths
            - 3 * mean * (sum2 / lengths)
            + 2 * mean ** 3
        )

        skew = np.divide(
            m3,
            m2 ** 1.5,
            out=np.zeros_like(m3),
            where=m2 > eps
        )

        return skew

    small_skew = _skewness(small_starts, small_ends)
    big_skew = _skewness(big_starts, big_ends)

    return small_skew - big_skew


# ========================================
# local_crest_factor_ratio
# ========================================
def local_crest_factor_ratio(signal, accuracy):
    """
    نسبت ضریب کرست (پیک به RMS) پنجره‌ی کوچک به پنجره‌ی بزرگ اطراف آن.

    نکته: پیک پنجره‌ی بزرگ به‌صورت تقریبی محاسبه می‌شود، از طریق ماکزیمم
    گرفتن روی چند بلوک همسایه (چون left و right دقیقاً معادل دو بلوک
    کوچک هستند). این تقریب برای بلوک‌های داخلی دقیق است؛ فقط در چند
    بلوک ابتدا و انتهای سیگنال ممکن است کران‌بندی کمی با نسخه‌ی
    نمونه‌به‌نمونه فرق کند.
    """

    if signal.ndim != 1:
        raise ValueError("سیگنال باید یک بعدی باشد")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal شامل مقادیر نامعتبر است")
    if accuracy not in range(1, 41):
        raise ValueError("دقت باید بین 1 تا 40 باشد")

    small_size = accuracy

    n = len(signal)

    small_starts = np.arange(0, n, small_size)
    small_ends = np.minimum(small_starts + small_size, n)

    big_starts = np.maximum(0, small_starts - small_size * 2)
    big_ends = np.minimum(small_ends + small_size * 2, n)

    power = signal ** 2
    cumsum_power = np.concatenate(([0], np.cumsum(power)))

    def _rms(starts, ends):
        lengths = ends - starts
        sums = cumsum_power[ends] - cumsum_power[starts]
        return np.sqrt(sums / lengths)

    small_rms = _rms(small_starts, small_ends)
    big_rms = _rms(big_starts, big_ends)

    abs_signal = np.abs(signal)
    block_max = np.maximum.reduceat(abs_signal, small_starts)

    n_blocks = len(block_max)
    padded = np.concatenate((
        np.full(2, -np.inf),
        block_max,
        np.full(2, -np.inf)
    ))

    big_max = np.full(n_blocks, -np.inf)
    for offset in range(5):
        big_max = np.maximum(big_max, padded[offset:offset + n_blocks])

    small_crest = np.divide(
        block_max,
        small_rms,
        out=np.zeros_like(block_max, dtype=float),
        where=small_rms != 0
    )

    big_crest = np.divide(
        big_max,
        big_rms,
        out=np.zeros_like(big_max, dtype=float),
        where=big_rms != 0
    )

    ratio = np.divide(
        small_crest,
        big_crest,
        out=np.zeros_like(small_crest),
        where=big_crest != 0
    )

    return ratio


import numpy as np

from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import rfft, rfftfreq


# ========================================
# ابزار مشترک دو تابع طیفی
# ========================================
def _local_windows(signal, accuracy):
    """
    استخراج پنجره‌های کوچک و بزرگ به‌صورت ماتریسی (batch)، برای FFT.

    تفاوت مهم با توابع قبلی: آنجا در لبه‌های سیگنال پنجره کوچک‌تر
    می‌شد (clipping). اینجا چون FFT به طول ثابت نیاز دارد، در لبه‌ها
    به‌جای کوچک‌شدن پنجره، با صفر پر می‌شود (zero-padding). این یک
    روش استاندارد و رایج در تحلیل طیفی پنجره‌ای است.
    """

    small_size = accuracy
    n = len(signal)

    pad_front = small_size * 2
    pad_back = small_size * 3
    big_size = small_size * 5

    padded = np.concatenate((
        np.zeros(pad_front),
        signal,
        np.zeros(pad_back)
    ))

    small_starts = np.arange(0, n, small_size)

    small_windows = sliding_window_view(padded, small_size)[small_starts + pad_front]
    big_windows = sliding_window_view(padded, big_size)[small_starts]

    return small_windows, big_windows


# ========================================
# local_spectral_centroid_ratio
# ========================================
def local_spectral_centroid_ratio(signal, fs, accuracy):
    """
    نسبت مرکز ثقل طیفی پنجره‌ی کوچک به پنجره‌ی بزرگ اطراف آن.

    مرکز ثقل طیفی، میانگین وزنی فرکانس است، وزن هر فرکانس همان
    دامنه‌ی طیف در آن فرکانس است. تغییر محتوای فرکانسی هنگام برخورد
    به عیب یا بک‌وال را نشان می‌دهد.
    """

    if signal.ndim != 1:
        raise ValueError("سیگنال باید یک بعدی باشد")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal شامل مقادیر نامعتبر است")
    if accuracy not in range(1, 41):
        raise ValueError("دقت باید بین 1 تا 40 باشد")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs باید یک عدد مثبت و معتبر باشد")

    small_windows, big_windows = _local_windows(signal, accuracy)

    freqs_small = rfftfreq(accuracy, d=1.0 / fs)
    freqs_big = rfftfreq(accuracy * 5, d=1.0 / fs)

    spec_small = np.abs(rfft(small_windows, axis=1))
    spec_big = np.abs(rfft(big_windows, axis=1))

    def _centroid(spec, freqs):
        weighted = spec @ freqs
        total = spec.sum(axis=1)
        return np.divide(
            weighted,
            total,
            out=np.zeros_like(total),
            where=total > 0
        )

    small_centroid = _centroid(spec_small, freqs_small)
    big_centroid = _centroid(spec_big, freqs_big)

    ratio = np.divide(
        small_centroid,
        big_centroid,
        out=np.zeros_like(small_centroid),
        where=big_centroid > 0
    )

    return ratio


# ========================================
# local_dominant_frequency_ratio
# ========================================
def local_dominant_frequency_ratio(signal, fs, accuracy):
    """
    نسبت فرکانس غالب (فرکانس متناظر با پیک طیف) پنجره‌ی کوچک به بزرگ.

    نسبت به مرکز ثقل طیفی ساده‌تر و ارزان‌تر است، اما چون فقط یک
    نمونه از طیف (پیک) را می‌بیند، به نویز حساس‌تر است.
    """

    if signal.ndim != 1:
        raise ValueError("سیگنال باید یک بعدی باشد")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal شامل مقادیر نامعتبر است")
    if accuracy not in range(1, 41):
        raise ValueError("دقت باید بین 1 تا 40 باشد")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs باید یک عدد مثبت و معتبر باشد")

    small_windows, big_windows = _local_windows(signal, accuracy)

    freqs_small = rfftfreq(accuracy, d=1.0 / fs)
    freqs_big = rfftfreq(accuracy * 5, d=1.0 / fs)

    spec_small = np.abs(rfft(small_windows, axis=1))
    spec_big = np.abs(rfft(big_windows, axis=1))

    small_dominant = freqs_small[np.argmax(spec_small, axis=1)]
    big_dominant = freqs_big[np.argmax(spec_big, axis=1)]

    ratio = np.divide(
        small_dominant,
        big_dominant,
        out=np.zeros_like(small_dominant),
        where=big_dominant > 0
    )

    return ratio


# ========================================
# local_entropy_ratio
# ========================================
def local_entropy_ratio(signal, accuracy):
    """
    نسبت آنتروپی شانون توزیع انرژی پنجره‌ی کوچک به پنجره‌ی بزرگ.

    توزیع انرژی یعنی هر نمونه با وزن (x_i^2 / انرژی کل پنجره) یک
    توزیع احتمال روی نمونه‌ها می‌سازد. اکوی منسجم، انرژی را در چند
    نمونه متمرکز می‌کند، پس آنتروپی پایین می‌آید؛ نویز پراکنده،
    آنتروپی بالایی دارد.

    برای اینکه این محاسبه هم مثل دو تابع اول شما با cumsum و بدون
    حلقه انجام شود، فرمول آنتروپی بازنویسی جبری شده است:

        entropy = -(1/S) * sum(x_i^2 * ln(x_i^2)) + ln(S)

    که در آن S مجموع انرژی (x_i^2) در همان پنجره است. این فرمول
    مستقیماً از تعریف آنتروپی شانون به‌دست می‌آید و چون بر پایه‌ی
    جمع‌های ساده است (نه میانگین‌های وابسته به هم)، جمع‌پذیر است و
    با همان ترفند cumsum قابل محاسبه است.
    """

    if signal.ndim != 1:
        raise ValueError("سیگنال باید یک بعدی باشد")
    if not np.all(np.isfinite(signal)):
        raise ValueError("signal شامل مقادیر نامعتبر است")
    if accuracy not in range(1, 41):
        raise ValueError("دقت باید بین 1 تا 40 باشد")

    small_size = accuracy
    left = accuracy * 2
    right = accuracy * 2

    n = len(signal)

    small_starts = np.arange(0, n, small_size)
    small_ends = np.minimum(small_starts + small_size, n)

    big_starts = np.maximum(0, small_starts - left)
    big_ends = np.minimum(small_ends + right, n)

    eps = np.finfo(float).eps

    power = signal ** 2
    log_power = np.where(power > eps, np.log(power), 0.0)
    g = power * log_power

    cumsum_power = np.concatenate(([0], np.cumsum(power)))
    cumsum_g = np.concatenate(([0], np.cumsum(g)))

    def _entropy(starts, ends):
        total_energy = cumsum_power[ends] - cumsum_power[starts]
        total_g = cumsum_g[ends] - cumsum_g[starts]

        entropy = np.zeros_like(total_energy)
        valid = total_energy > eps

        entropy[valid] = (
            -total_g[valid] / total_energy[valid]
            + np.log(total_energy[valid])
        )

        return np.maximum(entropy, 0)

    small_entropy = _entropy(small_starts, small_ends)
    big_entropy = _entropy(big_starts, big_ends)

    ratio = np.divide(
        small_entropy,
        big_entropy,
        out=np.zeros_like(small_entropy),
        where=big_entropy > eps
    )

    return ratio


# ========================================
# تست کردن
# ========================================



