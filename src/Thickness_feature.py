"""
بررسی همبستگی فیچرهای استخراج‌شده از A-scan با ضخامت واقعی نمونه‌ی Pk050
=========================================================================
این اسکریپت روی چند نقطه‌ی نمونه از هر ۴ پله‌ی ضخامت دیتاست BAM Pk050
اجرا می‌شود، ۹ فیچر معرفی‌شده را محاسبه می‌کند و همبستگی هرکدام را با
ضخامت واقعی (d_1..d_4) نشان می‌دهد.

قبل از اجرا، بخش "بارگذاری داده" را با فایل‌های واقعی خودت جایگزین کن.

نکات نسخه‌ی اصلاح‌شده (خلاصه):
- STEP_X_RANGE_MM اصلاح شد: مرزهای فیزیکی پله‌ها (0-500-1000-1500-2000)
  دیگر با نقطه‌ی شروع اندازه‌گیری (X=70mm) جابه‌جا نمی‌شوند.
- N_SAMPLES_PER_STEP واقعاً استفاده می‌شود (انتخاب تصادفی X با seed ثابت).
- خطاها دیگر بی‌صدا نادیده گرفته نمی‌شوند؛ در انتها گزارش می‌شوند.
- ابعاد داده و نرخ نمونه‌برداری قبل از اجرا اعتبارسنجی می‌شوند.

ساختار فایل (بازچینی‌شده، بدون تغییر منطق):
  1. بارگذاری داده
  2. مشخصات پله‌های ضخامت
  3. توابع کمکی سطح پایین (پنجره‌بندی، تشخیص سیگنال صفر)
  4. توابع استخراج فیچر (هرکدام یک فیچر/گروه فیچر)
  5. تابع تجمیع فیچرها (compute_all_features)
  6. اعتبارسنجی دیتاست
  7. رگرسیون
  8. اجرای اصلی (main)
"""

from pathlib import Path
import matplotlib as plt
import numpy as np
from scipy.signal import find_peaks, hilbert
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import LinearRegression, HuberRegressor

# =========================================================================
# 1. بارگذاری داده  --  این بخش را با فایل‌های واقعی خودت جایگزین کن
# =========================================================================
DATA_DIR = Path(r"C:\Users\asus\PycharmProjects\UltrasonicProject_ML\data")

data = np.load(DATA_DIR / "Pk050_3D_Dataset_Long_Rot00.npy", mmap_mode="r")
x_coords = np.load(DATA_DIR / "X-values.npy")
y_coords = np.load(DATA_DIR / "Y-values.npy")
time_us = np.load(DATA_DIR / "Z-values.npy")

SAMPLING_RATE_HZ = 2e6  # طبق توضیح دیتاست: 2 MS/s

# =========================================================================
# 2. مشخصات پله‌های ضخامت (طبق توضیح دیتاست BAM Pk050)
#    طول کل نمونه l = 2000 mm، ۴ پله‌ی فیزیکی با طول تقریبی ۵۰۰ mm هرکدام.
#    نقاط اندازه‌گیری‌شده از X≈70mm شروع می‌شوند، اما این نقطه‌ی شروعِ
#    اندازه‌گیری نیست که مرز پله‌های فیزیکی را تعیین می‌کند؛ مرزهای پله
#    بر اساس هندسه‌ی نمونه (هر پله ۵۰۰ mm) است. فقط نقاطی که واقعاً در
#    دیتاست اندازه‌گیری شده‌اند (که طبیعتاً از X≈70 شروع می‌شوند) پردازش
#    می‌شوند؛ داده‌ای برای X<70 وجود ندارد و ساخته نمی‌شود.
# =========================================================================
STEP_THICKNESS_MM = {1: 571.3, 2: 452.0, 3: 330.9, 4: 210.8}
STEP_X_RANGE_MM = {
    1: (0, 500),
    2: (500, 1000),
    3: (1000, 1500),
    4: (1500, 2000),
}
N_SAMPLES_PER_STEP = 8  # حداکثر چند موقعیت X تصادفی از هر پله نمونه‌گیری شود
RANDOM_SEED = 42  # برای تکرارپذیری انتخاب تصادفی موقعیت‌های X

FEATURE_NAMES = [
    "echo_period_us",
    "first_echo_time",
    "periodicity_strength",
    "periodicity_quality",
    "oscillation_ratio",
    "amplitude_decay_alpha",
    "dominant_frequency",
    "echo_count",
    "energy",
    "echo_spacing_precise",
    "echo_period_linfit",    
    "echo_period_linfit_r2",    
]

REQUIRED_FINITE_KEYS = [
    "echo_period_us",
    "periodicity_strength",
    "periodicity_quality",
    "oscillation_ratio",
    "dominant_frequency",
    "echo_count",
    "echo_period_linfit", 
]


# =========================================================================
# 3. توابع کمکی سطح پایین
# =========================================================================
def extract_window(signal, time_arr, center_time, half_window):
    """یک پنجره‌ی زمانی به مرکز و نیم‌عرض مشخص از سیگنال استخراج می‌کند."""
    mask = (time_arr >= center_time - half_window) & (time_arr <= center_time + half_window)
    return signal[mask], time_arr[mask]


def is_zero_signal(signal):
    """
    تشخیص نقاطی که هیچ سیگنال واقعی ندارند (خارج از سطح نمونه، پر شده با صفر).
    به‌جای np.allclose (که می‌تواند سیگنال‌های واقعی با دامنه‌ی خیلی کم را هم
    اشتباهاً صفر در نظر بگیرد)، اینجا فقط حالتی را «صفر» می‌دانیم که تمام
    مقادیر دقیقاً صفر باشند؛ این دقیقاً همان چیزی است که برای padding استفاده
    می‌شود، در حالی که نویز واقعی تقریباً هرگز دقیقاً صفر نیست.
    """
    return not np.any(signal)


# =========================================================================
# 4. توابع استخراج فیچر
# =========================================================================
def echo_periodicity_autocorr(
    signal,
    time,
    min_period=80,
    max_period=350,
    prominence_ratio=0.5,
    smooth_window=5,
):
    """
    Estimate echo periodicity using robust autocorrelation.

    Returns:
        period: estimated echo spacing
        strength: autocorrelation strength
        quality: confidence score
    """
    # 1) تبدیل ورودی
    signal = np.asarray(signal, dtype=float)
    time = np.asarray(time, dtype=float)

    # حذف پنجره‌ی پالس تحریک اولیه: این پالس خودش یک نوسان داخلی با
    # تناوب نزدیک به min_period دارد و بدون حذفش، اتوکورلیشن ممکن است
    # روی تناوب خودِ پالس (نه فاصله‌ی واقعی اکوها) قفل کند.
    exclude_before_us = 120.0
    keep_mask = time >= (time[0] + exclude_before_us)
    if np.count_nonzero(keep_mask) >= 10:
        signal = signal[keep_mask]
        time = time[keep_mask]

    if len(signal) < 10:
        raise ValueError("Signal too short")

    # 2) حذف مقادیر نامعتبر
    valid = np.isfinite(signal)
    signal = signal[valid]
    time = time[valid]

    # 3) فاصله‌ی نمونه‌برداری
    dt_values = np.diff(time)
    dt = np.median(dt_values)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Invalid dt")

    # 4) نرمال‌سازی سیگنال
    signal = signal - np.mean(signal)
    std = np.std(signal)
    if std == 0:
        raise ValueError("Zero energy signal")
    signal = signal / std

    # 5) هموارسازی اختیاری
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        signal = np.convolve(signal, kernel, mode="same")

    # 6) خودهمبستگی (autocorrelation)
    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr /= autocorr[0]

    # 7) تبدیل بازه‌ی تناوب به تعداد نمونه
    lag_min = int(min_period / dt)
    lag_max = int(max_period / dt)
    lag_max = min(lag_max, len(autocorr) - 1)
    if lag_max <= lag_min:
        raise ValueError("Invalid search range")
    search = autocorr[lag_min:lag_max]

    # 8) پیدا کردن پیک‌ها
    peaks, properties = find_peaks(search, prominence=0.05, distance=int(20 / dt))
    if len(peaks) == 0:
        best = np.argmax(search)
    else:
        values = search[peaks]
        max_peak = values.max()
        strong = peaks[values >= prominence_ratio * max_peak]
        best = strong[0] if len(strong) > 0 else peaks[np.argmax(values)]

    # 9) تناوب نهایی
    lag = lag_min + best
    period = lag * dt
    strength = autocorr[lag]

    # 10) فیچر کیفیت
    noise_level = np.std(autocorr[lag_min:lag])
    quality = strength / (noise_level + 1e-8)

    return period, strength, quality


def get_backwall_echoes(signal, time_arr, first_backwall, half_window=50,
                         max_echoes=4, search_margin=40):
    """
    اکوهای برگشتی متوالی را با شروع از تناوب تخمینی first_backwall پیدا
    می‌کند. هر اکویی که پیدا نشود یا خارج از بازه‌ی زمانی سیگنال باشد،
    باعث توقف حلقه می‌شود -- یعنی همیشه فرض نمی‌کند ۴ اکو وجود دارد،
    فقط همان‌هایی که واقعاً قابل‌تشخیص‌اند برمی‌گرداند.
    """
    if not np.isfinite(first_backwall) or first_backwall <= 0:
        raise ValueError("تناوب اکو (first_backwall) نامعتبر است.")

    echoes = {}
    t_max = time_arr[-1]
    for n in range(1, max_echoes + 1):
        approx_center = n * first_backwall
        if approx_center - (half_window + search_margin) > t_max:
            break  # این اکو و اکوهای بعدی خارج از بازه‌ی زمانی سیگنال‌اند

        search_sig, search_t = extract_window(
            signal, time_arr, approx_center, half_window + search_margin
        )
        if len(search_sig) == 0:
            break

        peak_idx = np.argmax(np.abs(search_sig))
        real_center = search_t[peak_idx]
        sig_win, t_win = extract_window(signal, time_arr, real_center, half_window)
        if len(t_win) == 0 or len(sig_win) == 0:
            break

        echoes[f"echo_{n}"] = (t_win, sig_win)
    return echoes


def oscillation_ratio(signal, time_arr, window_size=100, metric="ptp"):
    """نسبت نوسان پنجره‌ی اول به میانگین نوسان بقیه‌ی پنجره‌ها را محاسبه می‌کند."""
    if window_size <= 0:
        raise ValueError("window_size باید مثبت باشد.")

    signal = np.asarray(signal)
    time_arr = np.asarray(time_arr)
    t_start, t_end = time_arr[0], time_arr[-1]
    edges = np.arange(t_start, t_end + window_size, window_size)

    def compute_metric(segment):
        return np.ptp(segment) if metric == "ptp" else np.std(segment)

    window_values = []
    for i in range(len(edges) - 1):
        mask = (time_arr >= edges[i]) & (time_arr < edges[i + 1])
        segment = signal[mask]
        if segment.size > 0:
            window_values.append(compute_metric(segment))

    if len(window_values) < 2:
        raise ValueError("داده برای محاسبه‌ی حداقل دو بازه کافی نیست.")

    first_window_value = window_values[0]
    other_windows_mean = np.mean(window_values[1:])
    return first_window_value / other_windows_mean, first_window_value, other_windows_mean


def first_echo_time(signal, time, exclude_before_us=120.0):
    """
    زمان اولین اکو را از طریق پوش (envelope) هیلبرت تخمین می‌زند.
    بازه‌ی exclude_before_us از ابتدای سیگنال (خودِ پالس تحریک و دنباله‌ی
    ring-down آن) از جست‌وجو کنار گذاشته می‌شود، چون این پالس همیشه
    بزرگ‌ترین دامنه‌ی کل سیگنال را دارد و بدون این فیلتر، این تابع همیشه
    زمان پیکِ خودِ پالس تحریک را برمی‌گرداند -- نه backwall echo واقعی را.
    """
    signal = np.asarray(signal, dtype=float)
    time = np.asarray(time, dtype=float)
    mask = time >= (time[0] + exclude_before_us)
    if not np.any(mask):
        mask = np.ones_like(time, dtype=bool)  # fallback اگر سیگنال خیلی کوتاه بود

    analytic = hilbert(signal[mask])
    envelope = np.abs(analytic)
    idx = np.argmax(envelope)
    return time[mask][idx]

def _parabolic_peak(y, idx):
    """درون‌یابی سهمی اطراف یک پیک گسسته برای تخمین موقعیت زیرنمونه‌ای (sub-sample) پیک."""
    if idx <= 0 or idx >= len(y) - 1:
        return 0.0
    y0, y1, y2 = y[idx - 1], y[idx], y[idx + 1]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return 0.0
    return 0.5 * (y0 - y2) / denom

def echo_period_linfit(echoes, dt):
    """
    تخمین دقیق‌تر تناوب اکو با برازش خطی روی زمانِ (peak) تمام اکوهای
    شناسایی‌شده، به‌جای اتکا به فاصله‌ی یک جفت اکو یا یک پیک autocorrelation.
    این روش خطای تشخیص تک‌تک اکوها را با میانگین‌گیری روی چند نقطه کاهش
    می‌دهد و به‌طور فیزیکی معادل نرخ زمانِ رفت‌وبرگشت موج بین سطوح است.

    echoes: دیکشنری خروجی get_backwall_echoes -> {"echo_1": (t_win, sig_win), ...}
    dt: فاصله‌ی نمونه‌برداری (برای درون‌یابی سهمی دقیق‌تر پیک هر اکو)

    Returns:
        slope: تناوب اکوی برازش‌شده (میکروثانیه)
        r_squared: کیفیت برازش (هرچه به ۱ نزدیک‌تر، اکوها منظم‌ترند)
    """
    if len(echoes) < 2:
        raise ValueError("برای برازش خطی حداقل به ۲ اکو نیاز است.")

    echo_indices, arrival_times = [], []
    for key in sorted(echoes.keys(), key=lambda k: int(k.split("_")[1])):
        n = int(key.split("_")[1])
        t_win, sig_win = echoes[key]
        if len(sig_win) == 0:
            continue
        peak_idx = np.argmax(np.abs(sig_win))
        sub_shift = _parabolic_peak(np.abs(sig_win), peak_idx)
        arrival_time = t_win[peak_idx] + sub_shift * dt
        echo_indices.append(n)
        arrival_times.append(arrival_time)

    if len(echo_indices) < 2:
        raise ValueError("تعداد اکوهای معتبر برای برازش کافی نیست.")

    echo_indices = np.array(echo_indices, dtype=float)
    arrival_times = np.array(arrival_times, dtype=float)

    slope, intercept = np.polyfit(echo_indices, arrival_times, deg=1)

    fitted = slope * echo_indices + intercept
    ss_res = np.sum((arrival_times - fitted) ** 2)
    ss_tot = np.sum((arrival_times - arrival_times.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return slope, r_squared

def echo_spacing_precise(echoes, dt):
    """
    فاصله‌ی زمانی دقیق بین اکوی اول و دوم را با کراس-کورلیشن مستقیم بین
    دو پنجره‌ی اکو + درون‌یابی سهمی (sub-sample) محاسبه می‌کند.
    این فاصله مستقیماً با ضخامت نسبت دارد (d = v * spacing / 2) و چون
    از موقعیت واقعی دو اکوی تشخیص‌داده‌شده می‌آید (نه از خودهمبستگی کل
    سیگنال)، نسبت به echo_period_us دقیق‌تر و کم‌نویزتر است.
    """
    if "echo_1" not in echoes or "echo_2" not in echoes:
        raise ValueError("برای این فیچر حداقل به ۲ اکوی تشخیص‌داده‌شده نیاز است.")

    t1, s1 = echoes["echo_1"]
    t2, s2 = echoes["echo_2"]
    n = min(len(s1), len(s2))
    if n < 3:
        raise ValueError("پنجره‌ی اکوها برای کراس-کورلیشن خیلی کوتاه است.")

    s1 = np.asarray(s1[:n], dtype=float) - np.mean(s1[:n])
    s2 = np.asarray(s2[:n], dtype=float) - np.mean(s2[:n])

    corr = np.correlate(s2, s1, mode="full")
    peak_idx = np.argmax(corr)
    sub_shift = _parabolic_peak(corr, peak_idx)
    lag_samples = (peak_idx - (n - 1)) + sub_shift

    base_spacing = t2[0] - t1[0]  # فاصله‌ی مرکز دو پنجره به‌عنوان تخمین اولیه
    return base_spacing + lag_samples * dt

def echo_amplitude_decay(echoes):
    """نرخ افت دامنه‌ی اکوهای متوالی را با فیت خطی روی لگاریتم دامنه محاسبه می‌کند."""
    echo_indices, log_amplitudes, peak_amplitudes = [], [], []
    for i, key in enumerate(sorted(echoes.keys(), key=lambda k: int(k.split("_")[1])), start=1):
        _, sig_win = echoes[key]
        peak_amp = np.max(np.abs(sig_win))
        if peak_amp <= 0:
            continue
        echo_indices.append(i)
        peak_amplitudes.append(peak_amp)
        log_amplitudes.append(np.log(peak_amp))

    if len(echo_indices) < 2:
        raise ValueError("برای محاسبه‌ی نرخ افت، حداقل به ۲ اکو نیاز است.")

    echo_indices = np.array(echo_indices, dtype=float)
    log_amplitudes = np.array(log_amplitudes, dtype=float)
    slope, intercept = np.polyfit(echo_indices, log_amplitudes, 1)
    alpha = -slope
    fitted = slope * echo_indices + intercept
    ss_res = np.sum((log_amplitudes - fitted) ** 2)
    ss_tot = np.sum((log_amplitudes - np.mean(log_amplitudes)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return alpha, r_squared, peak_amplitudes


def dominant_frequency(signal, sampling_rate):
    """فرکانس غالب (بیشترین انرژی طیفی) کل سیگنال A-scan را برمی‌گرداند."""
    sig = np.asarray(signal, dtype=float)
    if sig.size == 0:
        return np.nan
    sig = sig - np.mean(sig)
    spectrum = np.abs(np.fft.rfft(sig))
    if spectrum.sum() == 0:
        return np.nan
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / sampling_rate)
    return freqs[np.argmax(spectrum)]


def echo_count_threshold(signal, time_arr, relative_threshold=0.15):
    """
    تعداد پیک‌های قابل‌تشخیص (اکو) که دامنه‌شان حداقل relative_threshold
    برابر بیشترین دامنه‌ی کل سیگنال است. هرچه نمونه نازک‌تر، اکوهای
    بیشتری در طول ثابت سیگنال جا می‌شوند.
    """
    sig = np.abs(np.asarray(signal, dtype=float))
    if sig.size == 0 or sig.max() == 0:
        return 0
    threshold = relative_threshold * sig.max()
    distance = max(1, len(sig) // 40)
    peaks, _ = find_peaks(sig, height=threshold, distance=distance)
    return len(peaks)


def signal_energy(signal):
    """انرژی سیگنال (مجموع مربعات نمونه‌ها)."""
    return np.sum(signal ** 2)


# =========================================================================
# 5. تجمیع همه‌ی فیچرها برای یک A-scan
# =========================================================================
def compute_all_features(signal, time_arr, sampling_rate):
    """۱۰ فیچر را برای یک A-scan محاسبه و در یک دیکشنری برمی‌گرداند."""
    period, strength, quality = echo_periodicity_autocorr(
        signal, time_arr, min_period=80, max_period=350
    )
    echoes = get_backwall_echoes(signal, time_arr, first_backwall=period)
    dt = np.median(np.diff(time_arr))
    osc_ratio, _, _ = oscillation_ratio(signal, time_arr)

    result = {
        "echo_period_us": period,
        "periodicity_strength": strength,
        "periodicity_quality": quality,
        "oscillation_ratio": osc_ratio,
        "dominant_frequency": dominant_frequency(signal, sampling_rate),
        "echo_count": echo_count_threshold(signal, time_arr),
        "backwall_time": period,
        "first_echo_time": first_echo_time(signal, time_arr),
        "energy": signal_energy(signal),
    }

    if len(echoes) >= 2:
        alpha, r2, _ = echo_amplitude_decay(echoes)
        result["amplitude_decay_alpha"] = alpha
        try:
            result["echo_spacing_precise"] = echo_spacing_precise(echoes, dt)
        except ValueError:
            result["echo_spacing_precise"] = np.nan
        try:
            period_lf, period_lf_r2 = echo_period_linfit(echoes, dt)
            result["echo_period_linfit"] = period_lf
            result["echo_period_linfit_r2"] = period_lf_r2
        except ValueError:
            result["echo_period_linfit"] = np.nan
            result["echo_period_linfit_r2"] = np.nan
    else:
        result["amplitude_decay_alpha"] = np.nan
        result["echo_spacing_precise"] = np.nan
        result["echo_period_linfit"] = np.nan
        result["echo_period_linfit_r2"] = np.nan

    return result


# =========================================================================
# 6. اعتبارسنجی دیتاست
# =========================================================================
def validate_dataset(data, x_coords, y_coords, time_us, sampling_rate_hz):
    """
    قبل از شروع پردازش، ابعاد داده و نرخ نمونه‌برداری را بررسی می‌کند.
    اگر چیزی ناسازگار باشد، به‌جای خطای گنگ بعداً، همین‌جا خطای واضح می‌دهد.
    """
    if data.ndim != 3:
        raise ValueError(f"انتظار می‌رفت data سه‌بعدی باشد، اما ndim={data.ndim} است.")
    if len(x_coords) != data.shape[0]:
        raise ValueError(
            f"طول x_coords ({len(x_coords)}) با data.shape[0] ({data.shape[0]}) مطابقت ندارد."
        )
    if len(y_coords) != data.shape[1]:
        raise ValueError(
            f"طول y_coords ({len(y_coords)}) با data.shape[1] ({data.shape[1]}) مطابقت ندارد."
        )
    if len(time_us) != data.shape[2]:
        raise ValueError(
            f"طول time_us ({len(time_us)}) با data.shape[2] ({data.shape[2]}) مطابقت ندارد."
        )

    dt_measured = np.mean(np.diff(time_us))
    dt_expected = 1e6 / sampling_rate_hz  # میکروثانیه، با فرض اینکه time_us به میکروثانیه است
    if not np.isclose(dt_measured, dt_expected, rtol=0.05):
        raise ValueError(
            f"فاصله‌ی زمانی اندازه‌گیری‌شده ({dt_measured:.4f} us) با فاصله‌ی مورد انتظار "
            f"({dt_expected:.4f} us بر اساس SAMPLING_RATE_HZ) همخوانی ندارد. "
            "احتمالاً واحد زمان یا نرخ نمونه‌برداری اشتباه است."
        )
    return dt_measured


# =========================================================================
# 7. رگرسیون
# =========================================================================
def run_regression_models(rows):
    """مدل Random Forest را با جست‌وجوی هایپرپارامتر (RandomizedSearchCV) روی
    GroupKFold (گروه‌بندی بر اساس موقعیت X) تنظیم و ارزیابی می‌کند."""
    X = np.array([[r[f] for f in FEATURE_NAMES] for r in rows], dtype=float)
    y = np.array([r["true_thickness_mm"] for r in rows], dtype=float)
    groups = np.array([r["x_idx"] for r in rows])

    valid = np.isfinite(X).all(axis=1)
    n_removed = int(len(X) - valid.sum())
    X, y, groups = X[valid], y[valid], groups[valid]

    print(f"نمونه‌های حذف‌شده به‌دلیل NaN/Inf در فیچرها: {n_removed}")
    print(f"نمونه‌های باقی‌مانده برای مدل‌سازی: {len(X)}\n")

    n_groups = len(np.unique(groups))
    if n_groups < 2:
        print("تعداد گروه‌های X (موقعیت‌های مکانی) برای اجرای GroupKFold کافی نیست.")
        return

    n_splits = min(5, n_groups)
    if n_splits < 5:
        print(
            f"توجه: تعداد گروه‌های X ({n_groups}) کمتر از ۵ است؛ "
            f"n_splits به {n_splits} کاهش یافت.\n"
        )

    gkf = GroupKFold(n_splits=n_splits)

    # فضای جست‌وجوی هایپرپارامتر برای Random Forest
    param_distributions = {
        "n_estimators": [300, 500, 800, 1200],
        "max_depth": [None, 8, 12, 16, 24],
        "min_samples_leaf": [1, 2, 3, 5, 8],
        "min_samples_split": [2, 4, 6, 10],
        "max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
    }

    base_model = RandomForestRegressor(random_state=42, n_jobs=1)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=40,
        scoring="neg_mean_absolute_error",
        cv=gkf,                 
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X, y, groups=groups)

    print("بهترین هایپرپارامترها:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    print()

    best_model = search.best_estimator_

    # ارزیابی نهایی با cross_val_predict روی همان تقسیم‌بندی GroupKFold،
    # با بهترین هایپرپارامترها (تا معیار MAE/RMSE/R2 مستقل از fold انتخاب‌شده در جست‌وجو باشد)
    preds = cross_val_predict(best_model, X, y, cv=gkf, groups=groups, n_jobs=-1)
    mae = mean_absolute_error(y, preds)
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
    r2 = r2_score(y, preds)

    print("Random Forest (تنظیم‌شده):")
    print(f"  MAE  = {mae:.2f} mm")
    print(f"  RMSE = {rmse:.2f} mm")
    print(f"  R2   = {r2:.3f}\n")

    best_model.fit(X, y)
    importances = sorted(
        zip(FEATURE_NAMES, best_model.feature_importances_),
        key=lambda t: t[1],
        reverse=True,
    )
    print("اهمیت فیچرها (نزولی):")
    for feat_name, imp in importances:
        print(f"  {feat_name}: {imp:.4f}")



# =========================================================================
# 7.5. تشخیص و ارزیابی عمیق‌تر مدل
# =========================================================================
def run_diagnostics(rows, output_dir=None):
    """
    سه تشخیص کلیدی را روی rows انجام می‌دهد:
      ۱) نمودار پراکندگی echo_period_us و echo_period_linfit در برابر ضخامت واقعی
      ۲) خطای مدل به تفکیک هر پله (روی همان GroupKFold مبتنی بر موقعیت X)
      ۳) ارزیابی Leave-One-Step-Out: تعمیم واقعی مدل به ضخامتی که اصلاً ندیده
    """
    import matplotlib.pyplot as plt

    X = np.array([[r[f] for f in FEATURE_NAMES] for r in rows], dtype=float)
    y = np.array([r["true_thickness_mm"] for r in rows], dtype=float)
    groups = np.array([r["x_idx"] for r in rows])
    steps = np.array([r["step"] for r in rows])

    valid = np.isfinite(X).all(axis=1)
    X, y, groups, steps = X[valid], y[valid], groups[valid], steps[valid]

    output_dir = Path(output_dir) if output_dir else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # ۱) نمودار پراکندگی: echo_period_us / echo_period_linfit در برابر ضخامت
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, feat_name in zip(axes, ["echo_period_us", "echo_period_linfit"]):
        idx = FEATURE_NAMES.index(feat_name)
        ax.scatter(y, X[:, idx], alpha=0.4, s=15)
        ax.set_xlabel("ضخامت واقعی (mm)")
        ax.set_ylabel(feat_name)
        ax.set_title(f"{feat_name} در برابر ضخامت واقعی")
        coeffs = np.polyfit(y, X[:, idx], deg=1)
        y_line = np.linspace(y.min(), y.max(), 100)
        ax.plot(y_line, np.polyval(coeffs, y_line), color="red", linewidth=1)
        corr = np.corrcoef(y, X[:, idx])[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, va="top")
    plt.tight_layout()
    scatter_path = output_dir / "diagnostic_scatter_period_vs_thickness.png"
    plt.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"[۱] نمودار پراکندگی ذخیره شد در: {scatter_path}")

    # ---------------------------------------------------------------
    # ۲) خطای مدل به تفکیک هر پله (GroupKFold روی موقعیت X، مثل قبل)
    # ---------------------------------------------------------------
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    base_model = RandomForestRegressor(
        n_estimators=800, max_depth=24, min_samples_split=2,
        min_samples_leaf=1, max_features=1.0, random_state=42, n_jobs=1,
    )
    preds_gkf = cross_val_predict(base_model, X, y, cv=gkf, groups=groups, n_jobs=-1)

    print("\n[۲] خطای مدل به تفکیک هر پله (GroupKFold روی موقعیت X):")
    for step in sorted(np.unique(steps)):
        mask = steps == step
        mae_step = mean_absolute_error(y[mask], preds_gkf[mask])
        bias_step = np.mean(preds_gkf[mask] - y[mask])
        print(
            f"  پله {step} (ضخامت واقعی={STEP_THICKNESS_MM[step]} mm, n={mask.sum()}): "
            f"MAE={mae_step:.2f} mm, میانگین بایاس={bias_step:+.2f} mm"
        )

    # ---------------------------------------------------------------
    # ۳) Leave-One-Step-Out: تعمیم واقعی به ضخامت کاملاً ندیده
    #    (آموزش روی ۳ پله، تست روی پله‌ی چهارمی که مدل هرگز ندیده)
    # ---------------------------------------------------------------
    print("\n[۳] ارزیابی Leave-One-Step-Out (تعمیم به ضخامت کاملاً ندیده):")
    loso_errors = []
    for test_step in sorted(np.unique(steps)):
        train_mask = steps != test_step
        test_mask = steps == test_step
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        model = RandomForestRegressor(
            n_estimators=800, max_depth=24, min_samples_split=2,
            min_samples_leaf=1, max_features=1.0, random_state=42, n_jobs=-1,
        )
        model.fit(X[train_mask], y[train_mask])
        preds_loso = model.predict(X[test_mask])

        mae_loso = mean_absolute_error(y[test_mask], preds_loso)
        mean_pred = preds_loso.mean()
        loso_errors.append(mae_loso)

        print(
            f"  آموزش روی پله‌های دیگر -> تست روی پله {test_step} "
            f"(ضخامت واقعی={STEP_THICKNESS_MM[test_step]} mm): "
            f"MAE={mae_loso:.2f} mm, میانگین پیش‌بینی={mean_pred:.1f} mm"
        )

    if loso_errors:
        print(f"\n  میانگین MAE در Leave-One-Step-Out: {np.mean(loso_errors):.2f} mm")
        print(
            "  (این عدد نشان می‌دهد مدل روی ضخامت کاملاً جدید چقدر خطا می‌دهد -- "
            "معیار واقعی‌تری از تعمیم‌پذیری نسبت به GroupKFold روی X در همان ۴ پله است.)"
        )

# =========================================================================
# 7.6. baseline خطی (برای مقایسه با RF در تعمیم به ضخامت ندیده)
# =========================================================================
def run_linear_baseline(rows, feature_name="echo_period_us"):
    """
    یک رگرسیون خطی ساده روی یک فیچر (پیش‌فرض echo_period_us) با ضخامت واقعی
    برازش می‌کند و همون ارزیابی GroupKFold + Leave-One-Step-Out را روی آن
    اجرا می‌کند. هدف: بررسی این‌که آیا مدل خطی -- که برخلاف RF می‌تواند
    خارج از بازه‌ی دیده‌شده هم extrapolate کند -- به ضخامت‌های کاملاً جدید
    بهتر تعمیم می‌دهد یا نه.
    """
    idx = FEATURE_NAMES.index(feature_name)
    X_all = np.array([[r[f] for f in FEATURE_NAMES] for r in rows], dtype=float)
    y = np.array([r["true_thickness_mm"] for r in rows], dtype=float)
    groups = np.array([r["x_idx"] for r in rows])
    steps = np.array([r["step"] for r in rows])

    valid = np.isfinite(X_all).all(axis=1)
    X_all, y, groups, steps = X_all[valid], y[valid], groups[valid], steps[valid]
    X = X_all[:, [idx]]

    print(f"\n=== Baseline خطی روی فیچر «{feature_name}» ===")

    # GroupKFold (مثل ارزیابی RF)
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    preds_gkf = cross_val_predict(LinearRegression(), X, y, cv=gkf, groups=groups)
    mae_gkf = mean_absolute_error(y, preds_gkf)
    r2_gkf = r2_score(y, preds_gkf)
    print(f"GroupKFold روی X: MAE={mae_gkf:.2f} mm, R2={r2_gkf:.3f}")

    # Leave-One-Step-Out
    print("Leave-One-Step-Out:")
    loso_errors = []
    for test_step in sorted(np.unique(steps)):
        train_mask = steps != test_step
        test_mask = steps == test_step
        model = LinearRegression()
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict(X[test_mask])
        mae_loso = mean_absolute_error(y[test_mask], preds)
        loso_errors.append(mae_loso)
        print(
            f"  train روی بقیه -> test پله {test_step} "
            f"(واقعی={STEP_THICKNESS_MM[test_step]} mm): "
            f"MAE={mae_loso:.2f} mm, میانگین پیش‌بینی={preds.mean():.1f} mm"
        )
    print(f"میانگین MAE در Leave-One-Step-Out (خطی): {np.mean(loso_errors):.2f} mm")

# =========================================================================
# 7.7. آمار فیچرهای کلیدی به تفکیک پله + سرعت ضمنی موج
# =========================================================================
def print_feature_stats_per_step(rows, feature_names=None):
    """
    برای هر پله، میانگین/انحراف‌معیار/میانه‌ی فیچرهای کلیدی و سرعت موج
    ضمنی (implied velocity = 2*d/period) را چاپ می‌کند. اگر سرعت ضمنی بین
    پله‌ها ناسازگار باشد یا پراکندگی زیادی داشته باشد، یعنی خودِ تخمین
    period به‌اندازه‌ی کافی دقیق/پایدار نیست.
    """
    if feature_names is None:
        feature_names = ["echo_period_us", "echo_period_linfit", "echo_spacing_precise"]

    print("\n=== آمار فیچرهای کلیدی به تفکیک پله ===")
    for step in sorted(STEP_THICKNESS_MM.keys()):
        step_rows = [r for r in rows if r["step"] == step]
        d = STEP_THICKNESS_MM[step]
        print(f"\nپله {step} (ضخامت واقعی={d} mm, n={len(step_rows)}):")
        for feat in feature_names:
            vals = np.array([r[feat] for r in step_rows], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                print(f"  {feat}: هیچ مقدار معتبری موجود نیست")
                continue
            velocity = 2000.0 * d / vals  # m/s ، از فرمول v = 2000*d(mm)/period(us)
            print(
                f"  {feat}: mean={vals.mean():.2f} us, std={vals.std():.2f} us, "
                f"median={np.median(vals):.2f} us, min={vals.min():.2f}, max={vals.max():.2f}  |  "
                f"implied v: mean={velocity.mean():.1f} m/s, std={velocity.std():.1f} m/s"
            )

# =========================================================================
# =========================================================================
# 7.8. رسم نمونه‌ی سیگنال‌ها به‌همراه period/اکوهای تشخیص‌داده‌شده
# =========================================================================
def plot_example_signals_per_step(rows, data, time_us, output_dir=".", n_examples=3):
    """
    برای هر پله، چند نمونه سیگنال خام (میانه، بیشترین و کمترین echo_period_us)
    را همراه با period تشخیص‌داده‌شده و موقعیت اکوهای backwall رسم می‌کند.
    هدف: چشمی بررسی کنیم آیا الگوریتم autocorrelation دارد روی اکوی درست
    قفل می‌کند یا روی هارمونیک/اکوی اشتباه.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for step in sorted(STEP_THICKNESS_MM.keys()):
        step_rows = [r for r in rows if r["step"] == step]
        if not step_rows:
            continue

        periods = np.array([r["echo_period_us"] for r in step_rows])
        order = np.argsort(periods)
        # سه نمونه: کمترین period، میانه، بیشترین period
        pick_idx = [order[0], order[len(order) // 2], order[-1]]
        picks = [step_rows[i] for i in pick_idx]

        fig, axes = plt.subplots(len(picks), 1, figsize=(10, 3 * len(picks)))
        if len(picks) == 1:
            axes = [axes]

        for ax, r in zip(axes, picks):
            signal = data[r["x_idx"], r["y_idx"], :]
            ax.plot(time_us, signal, linewidth=0.7)
            period = r["echo_period_us"]
            first_t = r["first_echo_time"]
            # چند تا اکوی مبتنی بر period تشخیص‌داده‌شده را با خط عمودی نشان بده
            for k in range(1, 5):
                t_echo = first_t + k * period
                if t_echo <= time_us.max():
                    ax.axvline(t_echo, color="red", linestyle="--", linewidth=0.8)
            ax.set_title(
                f"پله {step} (d={STEP_THICKNESS_MM[step]}mm) | "
                f"x_idx={r['x_idx']} y_idx={r['y_idx']} | "
                f"echo_period_us={period:.1f} first_echo={first_t:.1f}"
            )
            ax.set_xlabel("زمان (us)")

        plt.tight_layout()
        out_path = output_dir / f"diagnostic_signals_step{step}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"نمودار نمونه‌سیگنال‌های پله {step} ذخیره شد در: {out_path}")
# =========================================================================
# 8. اجرای اصلی
# =========================================================================
def main():
    dt_measured = validate_dataset(data, x_coords, y_coords, time_us, SAMPLING_RATE_HZ)

    print("=== اطلاعات دیتاست ===")
    print(f"شکل داده (X, Y, time): {data.shape}")
    print(f"بازه‌ی X: {x_coords.min():.1f} تا {x_coords.max():.1f} mm")
    print(f"بازه‌ی Y: {y_coords.min():.1f} تا {y_coords.max():.1f} mm")
    print(f"بازه‌ی زمان: {time_us.min():.2f} تا {time_us.max():.2f} us")
    print(f"فاصله‌ی نمونه‌برداری اندازه‌گیری‌شده: {dt_measured:.4f} us\n")

    rng = np.random.default_rng(RANDOM_SEED)

    rows = []
    failure_counts = {}

    for step, (x_lo, x_hi) in STEP_X_RANGE_MM.items():
        failure_counts[step] = {"ValueError": 0, "zero_signal": 0, "invalid_feature": 0}

        x_idx_candidates = np.where((x_coords >= x_lo) & (x_coords < x_hi))[0]
        if len(x_idx_candidates) == 0:
            print(f"هشدار: هیچ نقطه‌ی اندازه‌گیری‌شده‌ای در پله {step} (X: {x_lo}-{x_hi} mm) پیدا نشد.\n")
            continue

        n_select = min(N_SAMPLES_PER_STEP, len(x_idx_candidates))
        selected_x = rng.choice(x_idx_candidates, size=n_select, replace=False)
        selected_x.sort()

        n_y = data.shape[1]
        step_valid = 0

        for x_idx in selected_x:
            for y_idx in range(n_y):
                signal = data[x_idx, y_idx, :]

                if is_zero_signal(signal):
                    failure_counts[step]["zero_signal"] += 1
                    continue

                try:
                    feats = compute_all_features(signal, time_us, SAMPLING_RATE_HZ)
                except ValueError:
                    failure_counts[step]["ValueError"] += 1
                    continue

                if any(not np.isfinite(feats[k]) for k in REQUIRED_FINITE_KEYS):
                    failure_counts[step]["invalid_feature"] += 1
                    continue

                feats["true_thickness_mm"] = STEP_THICKNESS_MM[step]
                feats["step"] = step
                feats["x_idx"] = int(x_idx)
                feats["y_idx"] = y_idx
                rows.append(feats)
                step_valid += 1

        n_fail = sum(failure_counts[step].values())
        print(f"پله {step} (ضخامت واقعی: {STEP_THICKNESS_MM[step]} mm)")
        print(f"  موقعیت‌های X انتخاب‌شده: {len(selected_x)} از {len(x_idx_candidates)} موجود")
        print(f"  نمونه‌های معتبر: {step_valid}")
        print(
            f"  نمونه‌های ناموفق: {n_fail} "
            f"(ValueError={failure_counts[step]['ValueError']}, "
            f"zero_signal={failure_counts[step]['zero_signal']}, "
            f"invalid_feature={failure_counts[step]['invalid_feature']})\n"
        )

    total_valid = len(rows)
    total_failed = sum(sum(c.values()) for c in failure_counts.values())
    print(f"مجموع نمونه‌های معتبر: {total_valid}")
    print(f"مجموع نمونه‌های ناموفق: {total_failed}\n")

    if not rows:
        print("هیچ نمونه‌ای محاسبه نشد — بارگذاری داده یا آستانه‌ها را بررسی کن.")
        return

    print_feature_stats_per_step(rows)   
    
    plot_example_signals_per_step(rows, data, time_us)   # ← این خط جدید

    run_regression_models(rows)

    run_regression_models(rows)
    run_diagnostics(rows)
    run_linear_baseline(rows, feature_name="echo_period_us")
    run_linear_baseline(rows, feature_name="echo_period_linfit")


if __name__ == "__main__":
    main()