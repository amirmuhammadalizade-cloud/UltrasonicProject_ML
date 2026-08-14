import numpy as np
from scipy.ndimage import (
    gaussian_filter1d,
    median_filter,
    uniform_filter1d,
)
from scipy.signal import find_peaks, savgol_filter, hilbert
from dataclasses import dataclass, asdict
from scipy import signal as scipy_signal




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
            low: آستانه پایین به‌صورت کسری از دامنه پیک (پیش‌فرض ۰.۱).
            high: آستانه بالا به‌صورت کسری از دامنه پیک (پیش‌فرض ۰.۹).

        Returns:
            فاصله زمانی بین عبور صعودی از آستانه پایین و آستانه بالا،
            پیش از نمونه پیک. اگر یکی از آستانه‌ها یافت نشود صفر بازمی‌گردد.
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

        low_index = int(low_crossings[0])
        high_index = int(high_crossings[0])

        return float(self.time[high_index] - self.time[low_index])

    def signal_energy(self) -> float:
        return float(np.sum(self.signal ** 2))

    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.signal ** 2)))


class FrequencyDomainFeatures:
    """ویژگی‌های حوزه فرکانس (طیف FFT)."""

    def __init__(self, signal: np.ndarray, sampling_rate: float):
        self.signal = signal
        self.sampling_rate = sampling_rate

    def _magnitude_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """طیف دامنه یک‌طرفه FFT را محاسبه می‌کند (متد کمک داخلی).

        Returns:
            تاپلی از (فرکانس‌ها، دامنه‌های متناظر) برای فرکانس‌های غیرمنفی.
        """
        spectrum = np.fft.rfft(self.signal)
        frequencies = np.fft.rfftfreq(self.signal.size, d=1.0 / self.sampling_rate)
        magnitude = np.abs(spectrum)
        return frequencies, magnitude

    def dominant_frequency(self) -> float:
        """فرکانسی که بیشترین دامنه طیفی را دارد.

        Returns:
            فرکانس غالب بر حسب همان واحد sampling_rate (معمولاً هرتز).
        """
        frequencies, magnitude = self._magnitude_spectrum()
        dominant_index = int(np.argmax(magnitude))
        return float(frequencies[dominant_index])

    def spectral_centroid(self) -> float:
        """مرکز ثقل طیف (میانگین وزن‌دار فرکانس‌ها با وزن دامنه).

        Returns:
            فرکانس مرکز ثقل. اگر مجموع دامنه صفر باشد، صفر بازمی‌گردد.
        """
        frequencies, magnitude = self._magnitude_spectrum()
        total_magnitude = np.sum(magnitude)

        if total_magnitude == 0:
            return 0.0

        return float(np.sum(frequencies * magnitude) / total_magnitude)

    def bandwidth(self, level_db: float = -6.0) -> float:
        """پهنای باند سیگنال بر اساس آستانه سطح افت نسبت به پیک طیفی.

        Args:
            level_db: سطح افت نسبت به دامنه پیک بر حسب دسی‌بل (پیش‌فرض ۶-، معیار متداول در UT).

        Returns:
            پهنای باند (فرکانس بالا منهای فرکانس پایین) در محدوده‌ای که دامنه
            طیف بالاتر از سطح آستانه است. اگر پیک صفر باشد، صفر بازمی‌گردد.
        """
        frequencies, magnitude = self._magnitude_spectrum()
        peak_magnitude = np.max(magnitude)

        if peak_magnitude == 0:
            return 0.0

        threshold = peak_magnitude * (10 ** (level_db / 20.0))
        above_threshold = np.where(magnitude >= threshold)[0]

        if above_threshold.size == 0:
            return 0.0

        low_freq = frequencies[above_threshold[0]]
        high_freq = frequencies[above_threshold[-1]]

        return float(high_freq - low_freq)

    
@dataclass
class SignalFeatures:
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
    """یک نقطه ورودی که envelope/سیگنال فیلترشده رو می‌گیره و SignalFeatures برمی‌گردونه."""

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