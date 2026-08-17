"""
بررسی همبستگی فیچرهای استخراج‌شده از A-scan با ضخامت واقعی نمونه‌ی Pk050
=========================================================================
این اسکریپت روی چند نقطه‌ی نمونه از هر ۴ پله‌ی ضخامت دیتاست BAM Pk050
اجرا می‌شود، ۶ فیچر معرفی‌شده را محاسبه می‌کند و همبستگی هرکدام را با
ضخامت واقعی (d_1..d_4) نشان می‌دهد.

قبل از اجرا، بخش "بارگذاری داده" را با فایل‌های واقعی خودت جایگزین کن.

نکات نسخه‌ی اصلاح‌شده (خلاصه):
- STEP_X_RANGE_MM اصلاح شد: مرزهای فیزیکی پله‌ها (0-500-1000-1500-2000)
  دیگر با نقطه‌ی شروع اندازه‌گیری (X=70mm) جابه‌جا نمی‌شوند.
- N_SAMPLES_PER_STEP واقعاً استفاده می‌شود (انتخاب تصادفی X با seed ثابت).
- خطاها دیگر بی‌صدا نادیده گرفته نمی‌شوند؛ در انتها گزارش می‌شوند.
- ابعاد داده و نرخ نمونه‌برداری قبل از اجرا اعتبارسنجی می‌شوند.
"""

import numpy as np
from pathlib import Path
from scipy.signal import argrelextrema, find_peaks
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================================================
# 1. بارگذاری داده  --  این بخش را با فایل‌های واقعی خودت جایگزین کن
# =========================================================================
DATA_DIR = Path(r"C:\Users\asus\PycharmProjects\UltrasonicProject_ML\data")
data = np.load(
    DATA_DIR / "Pk050_3D_Dataset_Long_Rot00.npy",
    mmap_mode="r"
)

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
RANDOM_SEED = 42        # برای تکرارپذیری انتخاب تصادفی موقعیت‌های X
MIN_PERIODICITY_STRENGTH = 0.3  # نمونه‌هایی با تشخیص تناوب اکو ضعیف کنار گذاشته می‌شوند

# =========================================================================
# 3. توابع استخراج و فیچرها
# =========================================================================
def extract_window(signal, time_arr, center_time, half_window):
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


def echo_periodicity_autocorr(signal, time_arr, min_period=80, max_period=350,
                                prominence_ratio=0.5):
    """
    تناوب اکوهای برگشتی را از روی autocorrelation سیگنال تخمین می‌زند.
    به‌جای انتخاب بزرگ‌ترین پیک autocorrelation در کل بازه (که می‌تواند
    هارمونیک دوم/سوم period واقعی باشد)، اولین پیک محلی که نسبت به
    بزرگ‌ترین پیک بازه به‌اندازه‌ی کافی قوی است (prominence_ratio) را
    به‌عنوان تناوب بنیادین (fundamental) اکوها انتخاب می‌کند.

    max_period پیش‌فرض به ۳۵۰ افزایش یافته چون ۳۰۰ برای ضخیم‌ترین پله
    (d1 = 571.3 mm) ممکن است تناوب واقعی اکوها را قطع کند.
    """
    signal = np.asarray(signal, dtype=float)
    time_arr = np.asarray(time_arr, dtype=float)

    if signal.size < 4 or time_arr.size < 4:
        raise ValueError("سیگنال برای محاسبه‌ی autocorrelation خیلی کوتاه است.")

    dt = np.mean(np.diff(time_arr))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("فاصله‌ی زمانی بین نمونه‌ها (dt) نامعتبر است.")

    signal_zero_mean = signal - np.mean(signal)

    autocorr = np.correlate(signal_zero_mean, signal_zero_mean, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]

    if autocorr[0] == 0:
        raise ValueError("انرژی سیگنال صفر است؛ autocorrelation قابل نرمال‌سازی نیست.")
    autocorr = autocorr / autocorr[0]

    lag_min = int(min_period / dt)
    lag_max = int(max_period / dt)
    lag_max = min(lag_max, len(autocorr) - 1)

    if lag_min < 0 or lag_max <= lag_min:
        raise ValueError("بازه‌ی min_period/max_period با طول سیگنال سازگار نیست.")

    search_region = autocorr[lag_min:lag_max]
    if len(search_region) == 0:
        raise ValueError("بازه‌ی min_period/max_period با طول سیگنال سازگار نیست.")

    local_max_idx = argrelextrema(search_region, np.greater)[0]

    if len(local_max_idx) == 0:
        # fallback: هیچ پیک محلی واقعی پیدا نشد، از بزرگ‌ترین مقدار استفاده کن
        best_lag_rel = int(np.argmax(search_region))
    else:
        peak_vals = search_region[local_max_idx]
        threshold = prominence_ratio * peak_vals.max()
        strong_candidates = local_max_idx[peak_vals >= threshold]
        # کوچیک‌ترین lag در بین پیک‌های قوی = تناوب بنیادین (نه هارمونیک)
        best_lag_rel = int(strong_candidates[0])

    best_lag = lag_min + best_lag_rel
    period = best_lag * dt
    strength = autocorr[best_lag]

    if not np.isfinite(period) or period <= 0 or not np.isfinite(strength):
        raise ValueError("تناوب یا شدت محاسبه‌شده نامعتبر است.")

    return period, strength


def get_backwall_echoes(signal, time_arr, first_backwall, half_window=50,
                         max_echoes=3, search_margin=40):
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

        search_sig, search_t = extract_window(signal, time_arr, approx_center,
                                               half_window + search_margin)
        if len(search_sig) == 0:
            break

        peak_idx = np.argmax(np.abs(search_sig))
        real_center = search_t[peak_idx]
        sig_win, t_win = extract_window(signal, time_arr, real_center, half_window)
        if len(t_win) == 0 or len(sig_win) == 0:
            break

        echoes[f'echo_{n}'] = (t_win, sig_win)
    return echoes


def oscillation_ratio(signal, time_arr, window_size=100, metric='std'):
    if window_size <= 0:
        raise ValueError("window_size باید مثبت باشد.")

    signal = np.asarray(signal)
    time_arr = np.asarray(time_arr)
    t_start, t_end = time_arr[0], time_arr[-1]
    edges = np.arange(t_start, t_end + window_size, window_size)

    def compute_metric(segment):
        return np.ptp(segment) if metric == 'ptp' else np.std(segment)

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


def echo_amplitude_decay(echoes):
    echo_indices, log_amplitudes, peak_amplitudes = [], [], []
    for i, key in enumerate(sorted(echoes.keys(), key=lambda k: int(k.split('_')[1])), start=1):
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
    """فرکانس غالب (بیشترین انرژی طیفی) کل سیگنال A-scan."""
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


def compute_all_features(signal, time_arr, sampling_rate):
    """۶ فیچر معنادار را برای یک A-scan محاسبه و در یک دیکشنری برمی‌گرداند."""
    period, strength = echo_periodicity_autocorr(signal, time_arr,
                                                   min_period=80, max_period=350)
    echoes = get_backwall_echoes(signal, time_arr, first_backwall=period)

    osc_ratio, _, _ = oscillation_ratio(signal, time_arr)

    result = {
        'echo_period_us': period,          # قبلاً به‌اشتباه time_of_flight نامیده می‌شد
        'periodicity_strength': strength,
        'oscillation_ratio': osc_ratio,
    }
    result['dominant_frequency'] = dominant_frequency(signal, sampling_rate)
    result['echo_count'] = echo_count_threshold(signal, time_arr)

    if len(echoes) >= 2:
        alpha, r2, _ = echo_amplitude_decay(echoes)
        result['amplitude_decay_alpha'] = alpha
    else:
        result['amplitude_decay_alpha'] = np.nan

    return result


# =========================================================================
# 4. اعتبارسنجی دیتاست
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
# 5. رگرسیون
# =========================================================================
def run_regression_models(rows):
    feature_names = ['echo_period_us', 'periodicity_strength',
                      'oscillation_ratio', 'amplitude_decay_alpha',
                      'dominant_frequency', 'echo_count']

    X = np.array([[r[f] for f in feature_names] for r in rows], dtype=float)
    y = np.array([r['true_thickness_mm'] for r in rows], dtype=float)
    groups = np.array([r['x_idx'] for r in rows])

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
        print(f"توجه: تعداد گروه‌های X ({n_groups}) کمتر از ۵ است؛ "
              f"n_splits به {n_splits} کاهش یافت.\n")

    gkf = GroupKFold(n_splits=n_splits)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
    }

    for name, model in models.items():
        preds = cross_val_predict(model, X, y, cv=gkf, groups=groups)
        mae = mean_absolute_error(y, preds)
        rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
        r2 = r2_score(y, preds)
        print(f"{name}:")
        print(f"  MAE  = {mae:.2f} mm")
        print(f"  RMSE = {rmse:.2f} mm")
        print(f"  R2   = {r2:.3f}\n")

        # تشخیصی: اهمیت هر فیچر از دید Random Forest -- کمک می‌کنه بفهمیم کدوم
        # فیچر واقعاً مفیده. این مدل روی کل داده fit می‌شه، فقط برای دیدن اهمیت‌ها،
        # و در عدد MAE/RMSE/R2 بالا (که از cross-validation اومدن) شرکت نداره.
    rf_diagnostic = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    rf_diagnostic.fit(X, y)
    print("اهمیت فیچرها (Random Forest):")
    for fname, importance in sorted(zip(feature_names, rf_diagnostic.feature_importances_),
                                    key=lambda p: p[1], reverse=True):
        print(f"  {fname}: {importance:.3f}")


# =========================================================================
# 6. اجرای اصلی
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
        failure_counts[step] = {'ValueError': 0, 'zero_signal': 0, 'invalid_feature': 0}

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
                    failure_counts[step]['zero_signal'] += 1
                    continue

                try:
                    feats = compute_all_features(signal, time_us, SAMPLING_RATE_HZ)
                except ValueError:
                    failure_counts[step]['ValueError'] += 1
                    continue

                required_keys = ['echo_period_us', 'periodicity_strength',
                                  'oscillation_ratio', 'dominant_frequency', 'echo_count']
                if any(not np.isfinite(feats[k]) for k in required_keys):
                    failure_counts[step]['invalid_feature'] += 1
                    continue

                feats['true_thickness_mm'] = STEP_THICKNESS_MM[step]
                feats['step'] = step
                feats['x_idx'] = int(x_idx)
                feats['y_idx'] = y_idx
                rows.append(feats)
                step_valid += 1

        n_fail = sum(failure_counts[step].values())
        print(f"پله {step} (ضخامت واقعی: {STEP_THICKNESS_MM[step]} mm)")
        print(f"  موقعیت‌های X انتخاب‌شده: {len(selected_x)} از {len(x_idx_candidates)} موجود")
        print(f"  نمونه‌های معتبر: {step_valid}")
        print(f"  نمونه‌های ناموفق: {n_fail} "
              f"(ValueError={failure_counts[step]['ValueError']}, "
              f"zero_signal={failure_counts[step]['zero_signal']}, "
              f"invalid_feature={failure_counts[step]['invalid_feature']})\n")

    total_valid = len(rows)
    total_failed = sum(sum(c.values()) for c in failure_counts.values())
    print(f"مجموع نمونه‌های معتبر: {total_valid}")
    print(f"مجموع نمونه‌های ناموفق: {total_failed}\n")

    if not rows:
        print("هیچ نمونه‌ای محاسبه نشد — بارگذاری داده یا آستانه‌ها را بررسی کن.")
        return

    run_regression_models(rows)


if __name__ == "__main__":
    main()