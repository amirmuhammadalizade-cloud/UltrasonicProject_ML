"""پردازش سیگنال، استخراج ویژگی و برآورد عمق برای اندازه‌گیری‌های التراسونیک روی نمونه بتنی.

ساختار فایل (به ترتیب جریان داده):
    1. Signal / Filtering        -> دسترسی به داده خام و فیلترینگ سیگنال
    2. TimeDomainFeatures         -> ویژگی‌های حوزه زمان
    3. FrequencyDomainFeatures    -> ویژگی‌های حوزه فرکانس
    4. SignalFeatures / FeatureExtractor -> جمع‌آوری ویژگی‌ها در یک خروجی واحد
    5. EchoDetector               -> آشکارسازی چند echo در یک A-scan
    6. DepthEstimator             -> تبدیل زمان پرواز به عمق فیزیکی
    7. signal_to_noise_ratio      -> معیار کیفیت سیگنال
    8. DatasetBuilder             -> اجرای کل pipeline روی تمام نقاط شبکه
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import find_peaks, hilbert, savgol_filter


class Signal:
    """دسترسی به داده خام یک شبکه اندازه‌گیری و عملیات پایه روی هر A-scan."""

    def __init__(self, data: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.data = data
        self.x = x
        self.y = y
        self.z = z

    def non_empty_value_mask(self) -> np.ndarray:
        """ماسک نقاطی از شبکه که داده واقعی دارند (غیر صفر)."""
        return np.any(self.data != 0, axis=2)

    def compute_envelope(self, x_index: int, y_index: int) -> np.ndarray:
        """پوش (envelope) یک A-scan را با تبدیل هیلبرت محاسبه می‌کند."""
        mask = self.non_empty_value_mask()

        if not mask[x_index, y_index]:
            raise ValueError("Selected measurement point is empty.")

        a_scan = self.data[x_index, y_index, :]
        analytic_signal = hilbert(a_scan)
        return np.abs(analytic_signal)

    def Time(self) -> np.ndarray:
        """محور زمانی واقعی یک A-scan، بر حسب میکروثانیه.

        این محور از ``self.z`` (که در ``data_loader.load_files`` مستقیماً از
        فایل ``Z-values.npy`` بارگذاری شده) گرفته می‌شود، نه از یک گام زمانی
        فرضی و هارد-کدشده. استفاده از مقدار واقعیِ بارگذاری‌شده باعث می‌شود
        اگر گام نمونه‌برداری بین نمونه‌ها متفاوت باشد، محاسبات بعدی
        (time_of_flight، sampling_rate و غیره) همچنان درست بمانند.
        """
        return np.asarray(self.z, dtype=float)

    def set_filter(
        self, signal: np.ndarray, steps: list[tuple[str, dict]]
    ) -> "Filtering":
        """یک شیء Filtering با ترتیب و ترکیب دلخواه کاربر از فیلترها می‌سازد.

        Args:
            signal: سیگنال ورودی برای فیلتر کردن.
            steps: لیست ترتیبی از (نام_فیلتر, پارامترها)، مثلاً:
                [("median", {"size": 5}), ("gaussian", {"sigma": 4})]
        """
        return Filtering(signal=signal, steps=steps)


class Filtering:
    """زنجیره‌ای از فیلترها که به همان ترتیبی که مشخص شده اعمال می‌شوند."""

    _FILTER_FUNCTIONS = {
        "gaussian": gaussian_filter1d,
        "median": median_filter,
        "uniform": uniform_filter1d,
    }

    def __init__(self, signal: np.ndarray, steps: list[tuple[str, dict]]):
        """
        Args:
            signal: سیگنال ورودی.
            steps: لیست ترتیبی از (نام_فیلتر, پارامترها)، مثلاً:
                [("median", {"size": 5}), ("gaussian", {"sigma": 4})]
        """
        self.signal = signal
        self.steps = steps

    def apply(self) -> np.ndarray:
        signal = self.signal
        for name, params in self.steps:
            if name == "savgol":
                signal = savgol_filter(signal, **params)
            else:
                signal = self._FILTER_FUNCTIONS[name](signal, **params)
        return signal # type: ignore


class TimeDomainFeatures:
    """ویژگی‌های حوزه زمان یک A-scan (envelope یا سیگنال فیلتر شده)."""

    def __init__(self, signal: np.ndarray, time: np.ndarray):
        self.signal = signal
        self.time = time

    def peak_amplitude(self) -> float:
        return float(np.max(self.signal))

    def time_of_flight(self) -> float:
        peak_index = np.argmax(self.signal)
        return float(self.time[peak_index])

    def rise_time(self, low: float = 0.1, high: float = 0.9) -> float:
        """زمان صعود سیگنال بین دو آستانه نسبی (پیش‌فرض ۱۰٪ تا ۹۰٪ دامنه پیک).

        Args:
            low: آستانه پایین به‌صورت کسری از دامنه پیک.
            high: آستانه بالا به‌صورت کسری از دامنه پیک.

        Returns:
            فاصله زمانی بین عبور صعودی از آستانه پایین و بالا، پیش از پیک.
            اگر یکی از آستانه‌ها یافت نشود صفر بازمی‌گردد.
        """
        peak_index = int(np.argmax(self.signal))
        peak_value = self.signal[peak_index]

        if peak_value <= 0 or peak_index == 0:
            return 0.0

        low_level = low * peak_value
        high_level = high * peak_value
        segment = self.signal[: peak_index + 1]

        low_crossings = np.where(segment >= low_level)[0]
        high_crossings = np.where(segment >= high_level)[0]

        if low_crossings.size == 0 or high_crossings.size == 0:
            return 0.0

        return float(self.time[int(high_crossings[0])] - self.time[int(low_crossings[0])])

    def signal_energy(self) -> float:
        return float(np.sum(self.signal ** 2))

    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.signal ** 2)))


class FrequencyDomainFeatures:
    """ویژگی‌های حوزه فرکانس (طیف FFT)."""

    def __init__(self, signal: np.ndarray, sampling_rate: float):
        self.signal = signal
        self.sampling_rate = sampling_rate

    def _magnitude_spectrum(self) -> tuple:
        """طیف دامنه یک‌طرفه FFT (فرکانس‌ها، دامنه‌های متناظر)."""
        spectrum = np.fft.rfft(self.signal)
        frequencies = np.fft.rfftfreq(self.signal.size, d=1.0 / self.sampling_rate)
        magnitude = np.abs(spectrum)
        return frequencies, magnitude

    def dominant_frequency(self) -> float:
        frequencies, magnitude = self._magnitude_spectrum()
        return float(frequencies[int(np.argmax(magnitude))])

    def spectral_centroid(self) -> float:
        frequencies, magnitude = self._magnitude_spectrum()
        total_magnitude = np.sum(magnitude)

        if total_magnitude == 0:
            return 0.0

        return float(np.sum(frequencies * magnitude) / total_magnitude)

    def bandwidth(self, level_db: float = -6.0) -> float:
        """پهنای باند سیگنال بر اساس آستانه سطح افت نسبت به پیک طیفی (پیش‌فرض ۶- دسی‌بل)."""
        frequencies, magnitude = self._magnitude_spectrum()
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
    """خروجی ساخت‌یافته‌ی تمام ویژگی‌های استخراج‌شده از یک A-scan (آماده برای دیتاست ML)."""

    peak_amplitude: float
    time_of_flight: float
    rise_time: float
    signal_energy: float
    rms: float
    dominant_frequency: float
    spectral_centroid: float
    bandwidth: float

    def to_dict(self) -> dict:
        return asdict(self)


class FeatureExtractor:
    """نقطه ورودی واحد: envelope/سیگنال فیلترشده را می‌گیرد و SignalFeatures برمی‌گرداند."""

    def __init__(self, signal: np.ndarray, time: np.ndarray, sampling_rate: float):
        self.time_features = TimeDomainFeatures(signal, time)
        self.freq_features = FrequencyDomainFeatures(signal, sampling_rate)

    def extract(self) -> SignalFeatures:
        return SignalFeatures(
            peak_amplitude=self.time_features.peak_amplitude(),
            time_of_flight=self.time_features.time_of_flight(),
            rise_time=self.time_features.rise_time(),
            signal_energy=self.time_features.signal_energy(),
            rms=self.time_features.rms(),
            dominant_frequency=self.freq_features.dominant_frequency(),
            spectral_centroid=self.freq_features.spectral_centroid(),
            bandwidth=self.freq_features.bandwidth(),
        )


class EchoDetector:
    """آشکارسازی چندین echo در یک A-scan با استفاده از پیک‌یابی."""

    def __init__(self, signal: np.ndarray, time: np.ndarray):
        self.signal = signal
        self.time = time

    def find_echoes(self, min_height: float | None = None, min_distance: int = 20) -> tuple:
        """موقعیت زمانی و دامنه تمام echoهای قابل‌تشخیص را برمی‌گرداند.

        Returns:
            تاپلی از (اندیس نمونه‌ها، زمان echoها، دامنه echoها).
        """
        indices, _ = find_peaks(self.signal, height=min_height, distance=min_distance)
        return indices, self.time[indices], self.signal[indices]


class DepthEstimator:
    """تبدیل زمان پرواز موج به عمق فیزیکی نمونه (d = v * t / 2)."""

    def __init__(self, wave_velocity: float):
        """
        Args:
            wave_velocity: سرعت موج طولی در بتن، بر حسب متر بر ثانیه
                (معمولاً ۳۵۰۰ تا ۴۵۰۰ m/s).
        """
        self.wave_velocity = wave_velocity

    def estimate_depth(self, time_of_flight: float) -> float:
        """
        زمان پرواز رفت‌وبرگشت echo را به عمق فیزیکی تبدیل می‌کند.

        نکته واحدها: ``time_of_flight`` از ``Signal.Time()`` می‌آید و بر
        حسب میکروثانیه است، در حالی که ``wave_velocity`` بر حسب متر بر
        ثانیه است. پیش از استفاده در d = v * t / 2 باید زمان به ثانیه
        تبدیل شود، و چون خروجی مطلوب بر حسب میلی‌متر است (مطابق واحد
        سایر عمق‌ها در پروژه، از جمله ``DepthCalibration``)، نتیجه نهایی
        از متر به میلی‌متر هم تبدیل می‌شود.

        Args:
            time_of_flight: زمان پرواز رفت‌وبرگشت echo دیواره پشتی، بر
                حسب میکروثانیه.

        Returns:
            عمق تخمینی بر حسب میلی‌متر.
        """
        time_of_flight_s = time_of_flight * 1e-6
        depth_m = self.wave_velocity * time_of_flight_s / 2.0
        return depth_m * 1000.0