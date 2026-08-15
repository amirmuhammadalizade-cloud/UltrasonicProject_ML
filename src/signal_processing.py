import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import find_peaks, hilbert, savgol_filter
from dataclasses import asdict, dataclass


class Signal:
    def __init__(self, data, x, y, z, dead_zone_samples=90):
        self.data = data
        self.x = x
        self.y = y
        self.z = z
        self.dead_zone_samples = dead_zone_samples
        self.position_mask = np.any(self.data != 0, axis=2)
        self._envelope_volume = self._compute_envelope_volume()

    def time(self):
        return np.asarray(self.z[self.dead_zone_samples:], dtype=float)

    def raw_signal(self, x_index, y_index):
        return self.data[x_index, y_index, self.dead_zone_samples:]

    def _compute_envelope_volume(self) -> np.ndarray:
        """
        پوش سیگنال (envelope) را برای تمام نقاط معتبر گرید به‌صورت برداری
        (vectorized) و در یک فراخوانی واحد محاسبه می‌کند، به‌جای فراخوانی
        جداگانه‌ی hilbert برای هر پیکسل.

        محاسبه فقط روی نقاط معتبر (position_mask) انجام می‌شود تا هم در
        زمان محاسبه و هم در محاسبات بی‌مورد روی نقاط zero-padded صرفه‌جویی شود.

        Returns:
            آرایه‌ای هم‌شکل با self.data؛ در نقاط معتبر مقدار envelope و در
            نقاط خارج از ناحیه‌ی اندازه‌گیری مقدار صفر دارد.
        """
        envelope_volume = np.zeros_like(self.data, dtype=float)
        valid_scans = self.data[self.position_mask]  # shape: (n_valid, Z)
        envelope_volume[self.position_mask] = np.abs(hilbert(valid_scans, axis=1))
        return envelope_volume

    def compute_envelope(self, x_index: int, y_index: int) -> np.ndarray:
        return self._envelope_volume[x_index, y_index, self.dead_zone_samples:]


def apply_filters(signal, steps):
    for name, params in steps:
        if name == "savgol":
            signal = savgol_filter(signal, **params)
        elif name == "gaussian":
            signal = gaussian_filter1d(signal, **params)
        elif name == "median":
            signal = median_filter(signal, **params)
        elif name == "uniform":
            signal = uniform_filter1d(signal, **params)
        else:
            raise ValueError(f"Unknown filter step: {name!r}")
    return signal


def restrict_to_window(signal: np.ndarray, time: np.ndarray, t_min: float = 80.0, t_max: float = 320.0):
    """
    سیگنال و زمان را به بازه‌ی [t_min, t_max] محدود می‌کند تا از قفل‌شدن
    peak-detection روی ringdown اولیه‌ی مبدل (که تا حدود ۱۰۰۰µs ادامه دارد
    ولی همیشه بزرگ‌ترین دامنه‌ی مطلق سیگنال است) جلوگیری شود.

    بازه‌ی پیش‌فرض بر اساس محدوده‌ی زمان‌پرواز کالیبره‌شده‌ی قبلی
    (۱۱۳ تا ۲۷۳ میکروثانیه) با کمی حاشیه‌ی اطمینان انتخاب شده است.

    Args:
        signal: آرایه‌ی سیگنال (envelope یا raw) پس از حذف ناحیه‌ی مرده.
        time: آرایه‌ی زمان متناظر (میکروثانیه).
        t_min: ابتدای بازه‌ی مجاز جست‌وجوی اکو.
        t_max: انتهای بازه‌ی مجاز جست‌وجوی اکو.

    Returns:
        (windowed_signal, windowed_time)
    """
    mask = (time >= t_min) & (time <= t_max)
    return signal[mask], time[mask]


def find_echoes(signal, time, min_height=None, min_distance=20):
    indices, _ = find_peaks(signal, height=min_height, distance=min_distance)
    return indices, time[indices], signal[indices]


def strongest_echo(signal: np.ndarray, time: np.ndarray, min_distance: int = 20):
    """
    قوی‌ترین بیشینه‌ی محلی (echo) را در سیگنال پیدا می‌کند.

    برخلاف argmax خام که ممکن است لبه‌ی یک پنجره‌ی زمانی برش‌خورده را (که
    یک نقطه‌ی محلی واقعی نیست، بلکه فقط باقیمانده‌ی افت یک نوسان قبلی است)
    به‌اشتباه به‌عنوان پیک انتخاب کند، این تابع فقط بیشینه‌های محلی واقعی
    (نقاطی که همسایه‌های چپ و راستشان کمتر است) را در نظر می‌گیرد؛ نقاط
    ابتدا/انتهای بازه هرگز به‌عنوان پیک شناسایی نمی‌شوند.

    Args:
        signal: سیگنال (envelope) پس از محدودشدن به پنجره‌ی زمانی موردانتظار
            (خروجی restrict_to_window).
        time: آرایه‌ی زمان متناظر.
        min_distance: حداقل فاصله‌ی نمونه‌ای بین دو اکوی مجزا.

    Returns:
        (amplitude, time_of_flight) قوی‌ترین اکوی پیداشده. اگر هیچ بیشینه‌ی
        محلی واقعی پیدا نشود (سیگنال در کل پنجره یکنواخت صعودی/نزولی است)،
        به argmax ساده روی همان پنجره برمی‌گردد (fallback آشکار، نه صفر
        بی‌سروصدا).
    """
    indices, times, amplitudes = find_echoes(signal, time, min_distance=min_distance)
    if indices.size == 0:
        peak_index = int(np.argmax(signal))
        return float(signal[peak_index]), float(time[peak_index])
    best = int(np.argmax(amplitudes))
    return float(amplitudes[best]), float(times[best])


def peak_slope(signal: np.ndarray, time: np.ndarray) -> float:
    """
    شیب لبه‌ی صعودی سیگنال تا پیک اصلی (بیشینه‌ی کلی) را محاسبه می‌کند.

    الگوریتم:
    ۱. اندیس بیشینه‌ی کلی سیگنال (پیک اصلی) پیدا می‌شود.
    ۲. با جست‌وجوی رو به عقب از پیک -و فقط در همان بازه- نزدیک‌ترین کمینه‌ی
       محلی به‌عنوان نقطه‌ی شروع صعود انتخاب می‌شود. اگر چنین کمینه‌ای یافت
       نشود (سیگنال از همان نمونه‌ی اول در حال صعود بوده)، شروع صعود همان
       نمونه‌ی اول در نظر گرفته می‌شود.
    ۳. شیب به‌صورت (دامنه‌ی پیک - دامنه‌ی شروع) تقسیم بر (زمان پیک - زمان شروع)
       محاسبه می‌شود.

    محدود کردن جست‌وجو به بازه‌ی [۰, peak_index] باعث می‌شود شیب همیشه مربوط
    به لبه‌ی صعودی همان پیک اصلی باشد، نه پرشی نامرتبط به‌سمت یک اکوی دیگر.

    Args:
        signal: آرایه‌ی envelope (یا سیگنال فیلترشده) پس از حذف ناحیه‌ی مرده.
        time: آرایه‌ی زمان متناظر با signal (به میکروثانیه).

    Returns:
        شیب لبه‌ی صعودی تا پیک (دامنه بر میکروثانیه). در صورت نامعتبر بودن
        ورودی، یا اگر پیک در نمونه‌ی اول باشد، یا dt صفر/منفی شود، مقدار
        0.0 برگردانده می‌شود.
    """
    signal = np.asarray(signal, dtype=float)
    time = np.asarray(time, dtype=float)

    if signal.size < 2 or signal.size != time.size:
        return 0.0

    peak_index = int(np.argmax(signal))
    if peak_index == 0:
        return 0.0

    slopes_before_peak = np.diff(signal[: peak_index + 1])
    local_min_candidates = np.where(
        (slopes_before_peak[:-1] <= 0) & (slopes_before_peak[1:] > 0)
    )[0]

    start_index = (
        int(local_min_candidates[-1] + 1) if local_min_candidates.size > 0 else 0
    )

    dt = time[peak_index] - time[start_index]
    if dt <= 0:
        return 0.0

    return float((signal[peak_index] - signal[start_index]) / dt)


def signal_energy(signal):
    return float(np.sum(signal ** 2))


def rms(signal):
    return float(np.sqrt(np.mean(signal ** 2)))


def snr(signal: np.ndarray, noise_window: int = 500) -> float:
    """
    نسبت سیگنال به نویز (SNR) را با مقایسه‌ی دامنه‌ی پیک اصلی به انحراف‌معیار
    یک بازه‌ی اولیه (فرض بر نویز زمینه پیش از رسیدن اکوی برگشتی) محاسبه می‌کند.

    مهم: این تابع باید روی سیگنال کامل (پیش از restrict_to_window) صدا زده
    شود، چون بازه‌ی نویز مرجع از ابتدای سیگنال گرفته می‌شود.

    Args:
        signal: سیگنال (envelope یا raw) پس از حذف ناحیه‌ی مرده، پیش از
            محدودشدن به پنجره‌ی زمانی.
        noise_window: تعداد نمونه‌ی ابتدایی که به‌عنوان نویز زمینه در نظر
            گرفته می‌شود.

    Returns:
        نسبت دامنه‌ی پیک به انحراف‌معیار نویز زمینه. در صورت صفر بودن نویز
        زمینه یا سیگنال کوتاه‌تر از noise_window، مقدار 0.0 برمی‌گردد.
    """
    if signal.size <= noise_window:
        return 0.0
    noise_std = float(np.std(signal[:noise_window]))
    if noise_std == 0:
        return 0.0
    return float(np.max(signal) / noise_std)


def magnitude_spectrum(signal, sampling_rate):
    spectrum = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sampling_rate)
    magnitude = np.abs(spectrum)
    return frequencies, magnitude


def dominant_frequency(signal, sampling_rate):
    frequencies, magnitude = magnitude_spectrum(signal, sampling_rate)
    return float(frequencies[int(np.argmax(magnitude))])


def spectral_centroid(signal, sampling_rate):
    frequencies, magnitude = magnitude_spectrum(signal, sampling_rate)
    total_magnitude = np.sum(magnitude)

    if total_magnitude == 0:
        return 0.0

    return float(np.sum(frequencies * magnitude) / total_magnitude)


def bandwidth(signal, sampling_rate, level_db=-6.0):
    frequencies, magnitude = magnitude_spectrum(signal, sampling_rate)
    peak_magnitude = np.max(magnitude)

    if peak_magnitude == 0:
        return 0.0

    threshold = peak_magnitude * (10 ** (level_db / 20.0))
    above_threshold = np.where(magnitude >= threshold)[0]

    if above_threshold.size == 0:
        return 0.0

    return float(frequencies[above_threshold[-1]] - frequencies[above_threshold[0]])


@dataclass
class SignalFeatures:
    peak_amplitude: float
    time_of_flight: float
    peak_slope: float
    signal_energy: float
    rms: float
    dominant_frequency: float
    spectral_centroid: float
    bandwidth: float

    def to_dict(self):
        return asdict(self)


def extract_features(envelope, raw_signal, time, sampling_rate):
    peak_amp, tof = strongest_echo(envelope, time)
    return SignalFeatures(
        peak_amplitude=peak_amp,
        time_of_flight=tof,
        peak_slope=peak_slope(envelope, time),
        signal_energy=signal_energy(envelope),
        rms=rms(envelope),
        dominant_frequency=dominant_frequency(raw_signal, sampling_rate),
        spectral_centroid=spectral_centroid(raw_signal, sampling_rate),
        bandwidth=bandwidth(raw_signal, sampling_rate),
    )


def groundtruth_depth(x_position_mm: float) -> float:
    """
    ضخامت واقعی نمونه‌ی پله‌ای Pk050 را بر اساس موقعیت X (به میلی‌متر) برمی‌گرداند.

    نمونه شامل ۴ پله با طول تقریبی ۵۰۰ میلی‌متر و ضخامت‌های مشخص است. این تابع
    فقط یک جدول جست‌وجوی ساده بر اساس مرز پله‌هاست، نه یک مدل فیزیکی.
    ترتیب پله‌ها (X کوچک = ضخیم‌ترین) با بررسی همبستگی time_of_flight نسبت
    به x_position_mm در دیتاست واقعی تأیید شده است.

    Args:
        x_position_mm: موقعیت واقعی نقطه روی محور X به میلی‌متر (نه اندیس گرید).

    Returns:
        ضخامت واقعی نمونه در همان موقعیت (میلی‌متر).

    Raises:
        ValueError: اگر x_position_mm خارج از بازه‌ی معتبر [0, 2000] باشد.
    """
    if not 0 <= x_position_mm <= 2000:
        raise ValueError(f"x_position_mm خارج از بازه‌ی معتبر است: {x_position_mm}")

    if x_position_mm < 500:
        return 571.3
    elif x_position_mm < 1000:
        return 452.0
    elif x_position_mm < 1500:
        return 330.9
    else:
        return 210.8


def estimate_depth(time_of_flight_us, wave_velocity):
    time_of_flight_s = time_of_flight_us * 1e-6
    depth_m = wave_velocity * time_of_flight_s / 2.0
    return depth_m * 1000.0