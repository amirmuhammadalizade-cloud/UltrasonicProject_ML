"""
features.py
===========
توابع استخراج فیچر از یک سیگنال A-scan، به‌همراه تابع نهایی
`compute_all_features` که همه‌ی فیچرها را برای یک سیگنال محاسبه و در
قالب یک دیکشنری برمی‌گرداند.

این فایل معادل بخش‌های ۴ و ۵ اسکریپت اصلی است:
  4. توابع استخراج فیچر (هرکدام یک فیچر/گروه فیچر)
  5. تابع تجمیع فیچرها (compute_all_features)

برای پنجره‌بندی سیگنال، تشخیصِ صفر بودن، پیدا کردنِ اکوهای برگشتی و
درون‌یابیِ سهمیِ پیک‌ها، این فایل از توابعِ کمکیِ signal_utils.py
استفاده می‌کند.
"""

import numpy as np
from scipy.signal import find_peaks, hilbert

from signal_utils import extract_window, get_backwall_echoes, _parabolic_peak


def echo_periodicity_autocorr(
    signal,
    time,
    min_period=80,
    max_period=350,
    prominence_ratio=0.5,
    smooth_window=5,
):
    """
    تخمینِ تناوبِ اکوها (echo periodicity) با استفاده از خودهمبستگیِ
    (autocorrelation) مقاوم.

    مراحلِ کار (شماره‌گذاری‌شده در کد، برای پیگیریِ راحت‌تر):
        1) تبدیل ورودی به آرایه‌ی float
        2) حذفِ پنجره‌ی پالسِ تحریکِ اولیه
        3) حذفِ مقادیرِ نامعتبر (NaN/Inf)
        4) محاسبه‌ی فاصله‌ی نمونه‌برداری dt
        5) نرمال‌سازیِ سیگنال (میانگین صفر، انحراف‌معیار واحد)
        6) هموارسازیِ اختیاری (میانگین‌گیریِ متحرک)
        7) محاسبه‌ی خودهمبستگی
        8) تبدیلِ بازه‌ی تناوبِ فیزیکی (min_period..max_period) به بازه‌ی
           نمونه (lag)
        9) پیدا کردنِ پیک‌های خودهمبستگی در آن بازه
        10) انتخابِ تناوبِ نهایی و محاسبه‌ی امتیازِ کیفیت

    پارامترها:
        signal, time      : سیگنال و محور زمانِ آن
        min_period        : کمترین تناوبِ فیزیکاً معقول (میکروثانیه)
        max_period        : بیشترین تناوبِ فیزیکاً معقول (میکروثانیه)
        prominence_ratio  : آستانه‌ی نسبیِ انتخابِ پیکِ خودهمبستگی —
                             هر پیکی با ارتفاعِ حداقل prominence_ratio
                             برابرِ بلندترین پیک، به‌عنوانِ پیکِ اصلی
                             پذیرفته می‌شود (نه لزوماً بلندترین پیک).
        smooth_window     : طولِ پنجره‌ی هموارسازیِ اختیاری روی سیگنال

    خروجی:
        period   : تناوبِ تخمینیِ اکوها (میکروثانیه)
        strength : مقدارِ خودهمبستگی در آن تناوب (شاخصِ قدرتِ تناوب)
        quality  : امتیازِ کیفیت/اطمینان (strength نسبت‌به سطحِ نویزِ اطراف)
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

    # 10) فیچر کیفیت — هرچه strength نسبت به نوسانِ اطرافِ آن (نویز) بزرگ‌تر
    # باشد، اطمینانِ بیشتری به تناوبِ تشخیص‌داده‌شده داریم.
    noise_level = np.std(autocorr[lag_min:lag])
    quality = strength / (noise_level + 1e-8)

    return period, strength, quality


def oscillation_ratio(signal, time_arr, window_size=100, metric="ptp"):
    """
    نسبتِ میزانِ نوسانِ پنجره‌ی اولِ سیگنال (که معمولاً شاملِ پالسِ
    تحریک است) به میانگینِ نوسانِ بقیه‌ی پنجره‌ها را محاسبه می‌کند.

    این نسبت می‌تواند شاخصی از میزانِ افتِ سریعِ دامنه بعد از پالسِ
    اولیه باشد (مرتبط با میرایی/جذبِ موج در نمونه).

    پارامترها:
        signal, time_arr : سیگنال و محور زمانِ آن
        window_size      : طولِ هر پنجره (همان واحدِ زمانیِ time_arr)
        metric            : "ptp" (peak-to-peak) یا هر مقدارِ دیگر که در
                             این صورت انحرافِ‌معیار (std) استفاده می‌شود

    خروجی:
        (ratio, first_window_value, other_windows_mean)
        ratio: نسبتِ نوسانِ پنجره‌ی اول به میانگینِ نوسانِ بقیه

    خطا:
        اگر window_size نامعتبر باشد یا داده برای حداقل دو پنجره کافی
        نباشد، ValueError صادر می‌کند.
    """
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
    زمانِ اولین اکو را از طریقِ پوشِ (envelope) هیلبرت تخمین می‌زند.

    چرا exclude_before_us لازم است؟
    -------------------------------------------------------------------
    بازه‌ی exclude_before_us از ابتدای سیگنال (خودِ پالسِ تحریک و دنباله‌ی
    ring-down آن) از جست‌وجو کنار گذاشته می‌شود، چون این پالس همیشه
    بزرگ‌ترین دامنه‌ی کلِ سیگنال را دارد. بدون این فیلتر، این تابع همیشه
    زمانِ پیکِ خودِ پالسِ تحریک را برمی‌گرداند — نه backwall echo واقعی را.

    پارامترها:
        signal, time      : سیگنال و محور زمانِ آن
        exclude_before_us : بازه‌ای از ابتدای سیگنال که نادیده گرفته می‌شود

    خروجی:
        زمانِ تخمینیِ اولین اکو (همان واحدِ time)
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


def echo_period_linfit(echoes, dt):
    """
    تخمینِ دقیق‌ترِ تناوبِ اکو با برازشِ خطی روی زمانِ (peak) تمامِ اکوهای
    شناسایی‌شده، به‌جای اتکا به فاصله‌ی فقط یک جفت اکو یا یک پیکِ
    autocorrelation.

    این روش خطای تشخیصِ تک‌تکِ اکوها را با میانگین‌گیری روی چند نقطه
    کاهش می‌دهد و به‌طور فیزیکی معادلِ نرخِ زمانِ رفت‌وبرگشتِ موج بینِ
    سطوح است.

    پارامترها:
        echoes : دیکشنریِ خروجیِ get_backwall_echoes
                 -> {"echo_1": (t_win, sig_win), ...}
        dt     : فاصله‌ی نمونه‌برداری (برای درون‌یابیِ سهمیِ دقیق‌ترِ پیکِ
                 هر اکو)

    خروجی:
        slope     : تناوبِ اکوی برازش‌شده (میکروثانیه)
        r_squared : کیفیتِ برازش (هرچه به ۱ نزدیک‌تر، اکوها منظم‌ترند)

    خطا:
        اگر تعدادِ اکوهای معتبر کمتر از ۲ باشد، ValueError صادر می‌کند.
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
    فاصله‌ی زمانیِ دقیق بینِ اکوی اول و دوم را با کراس-کورلیشنِ مستقیم
    بینِ دو پنجره‌ی اکو + درون‌یابیِ سهمی (sub-sample) محاسبه می‌کند.

    این فاصله مستقیماً با ضخامت نسبت دارد (d = v * spacing / 2) و چون
    از موقعیتِ واقعیِ دو اکویِ تشخیص‌داده‌شده می‌آید (نه از خودهمبستگیِ
    کلِ سیگنال)، نسبت به echo_period_us دقیق‌تر و کم‌نویزتر است.

    پارامترها:
        echoes : دیکشنریِ خروجیِ get_backwall_echoes (باید حداقل شامل
                 echo_1 و echo_2 باشد)
        dt     : فاصله‌ی نمونه‌برداری

    خروجی:
        فاصله‌ی زمانیِ دقیقِ بینِ اکوی اول و دوم (همان واحدِ زمانیِ echoes)

    خطا:
        اگر echo_1 یا echo_2 در echoes نباشند، یا پنجره‌ی اکوها برای
        کراس-کورلیشن خیلی کوتاه باشد، ValueError صادر می‌کند.
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
    """
    نرخِ افتِ دامنه‌ی اکوهای متوالی را با فیتِ خطی روی لگاریتمِ دامنه
    محاسبه می‌کند (مدلِ افتِ نمایی: amplitude ~ exp(-alpha * n)).

    پارامترها:
        echoes : دیکشنریِ خروجیِ get_backwall_echoes

    خروجی:
        alpha           : نرخِ افتِ دامنه (هرچه بزرگ‌تر، افتِ سریع‌تر)
        r_squared       : کیفیتِ برازشِ خطی روی لگاریتمِ دامنه‌ها
        peak_amplitudes : لیستِ دامنه‌ی پیکِ هر اکو (قبل از لگاریتم‌گیری)

    خطا:
        اگر تعدادِ اکوهایی که دامنه‌ی مثبت دارند کمتر از ۲ باشد،
        ValueError صادر می‌کند.
    """
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
    """
    فرکانسِ غالب (بیشترین انرژیِ طیفی) کلِ سیگنالِ A-scan را با استفاده
    از تبدیلِ فوریه (FFT) برمی‌گرداند.

    پارامترها:
        signal         : آرایه‌ی دامنه‌ی سیگنال
        sampling_rate  : نرخِ نمونه‌برداری (هرتز)

    خروجی:
        فرکانسِ غالب (هرتز)، یا np.nan اگر سیگنال خالی باشد یا انرژیِ
        طیفیِ آن صفر باشد.
    """
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
    تعدادِ پیک‌های قابلِ‌تشخیص (اکو) را می‌شمارد؛ پیکی «قابلِ‌تشخیص»
    است که دامنه‌اش حداقل relative_threshold برابرِ بیشترین دامنه‌ی
    کلِ سیگنال باشد.

    هرچه نمونه نازک‌تر باشد، اکوهای بیشتری در طولِ ثابتِ سیگنال جا
    می‌شوند؛ پس این فیچر می‌تواند به‌طورِ غیرمستقیم با ضخامت رابطه
    داشته باشد.

    پارامترها:
        signal, time_arr    : سیگنال و محور زمانِ آن
        relative_threshold  : آستانه‌ی نسبیِ ارتفاعِ پیک (نسبت به
                               بیشترین دامنه‌ی سیگنال)

    خروجی:
        تعدادِ پیک‌های تشخیص‌داده‌شده (عدد صحیح). اگر سیگنال خالی یا
        کاملاً صفر باشد، 0 برمی‌گرداند.
    """
    sig = np.abs(np.asarray(signal, dtype=float))
    if sig.size == 0 or sig.max() == 0:
        return 0
    threshold = relative_threshold * sig.max()
    distance = max(1, len(sig) // 40)
    peaks, _ = find_peaks(sig, height=threshold, distance=distance)
    return len(peaks)


def signal_energy(signal):
    """انرژیِ سیگنال (مجموعِ مربعاتِ نمونه‌ها)."""
    return np.sum(signal ** 2)


# =========================================================================
# 5. تجمیع همه‌ی فیچرها برای یک A-scan
# =========================================================================
def compute_all_features(signal, time_arr, sampling_rate):
    """
    تمامِ فیچرهای تعریف‌شده در این فایل را برای یک سیگنالِ A-scan
    محاسبه می‌کند و در قالبِ یک دیکشنری برمی‌گرداند.

    ترتیبِ کار:
        1) تخمینِ تناوبِ اکو با autocorrelation (echo_periodicity_autocorr)
        2) پیدا کردنِ اکوهای واقعی بر اساسِ همان تناوب (get_backwall_echoes)
        3) محاسبه‌ی بقیه‌ی فیچرهای مستقل (oscillation_ratio,
           dominant_frequency, echo_count, first_echo_time, energy)
        4) اگر حداقل ۲ اکو پیدا شده باشد، فیچرهای وابسته به چند اکو را
           هم اضافه می‌کند (amplitude_decay_alpha, echo_spacing_precise,
           echo_period_linfit, echo_period_linfit_r2)؛ در غیرِ این صورت
           این فیچرها با مقدارِ NaN پر می‌شوند تا طولِ دیکشنریِ خروجی
           همیشه یکسان بماند.

    پارامترها:
        signal, time_arr : سیگنال و محور زمانِ آن
        sampling_rate    : نرخِ نمونه‌برداری (هرتز)

    خروجی:
        دیکشنریِ فیچرها، با کلیدهایی مطابقِ FEATURE_NAMES در config.py
        (به‌علاوه‌ی دو کلیدِ کمکیِ backwall_time که همان period است).

    خطا:
        اگر echo_periodicity_autocorr نتواند تناوبِ معتبری تخمین بزند
        (مثلاً سیگنالِ خیلی کوتاه یا بی‌انرژی)، ValueError صادر می‌کند و
        این استثنا تا سطحِ main() بالا می‌رود (که آنجا مدیریت می‌شود).
    """
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