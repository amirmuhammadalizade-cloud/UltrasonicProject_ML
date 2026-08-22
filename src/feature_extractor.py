"""
feature_extractor.py
=====================
کلاس SignalFeatures خروجی تمام توابع استخراج ویژگی موجود در
features.py را برای یک سیگنال محاسبه و در خودش نگه می‌دارد:

    - self.raw     : خروجی خام هر تابع (برخی عدد، برخی آرایه)
    - self.summary : آماره‌های خلاصه با طول ثابت (mean/std/min/max)
                      که برای مدل‌های یادگیری ماشین (مثل Random Forest)
                      مناسب است، چون طول signal بین نمونه‌ها می‌تواند
                      فرق کند اما طول summary همیشه ثابت است.

نکته: چون local_power_ratio ، local_std_ratio و ... برای هر سیگنال
یک آرایه (به تعداد بلوک‌ها) برمی‌گردانند، نمی‌توان مستقیم آن‌ها را
ستون یک جدول ویژگی کرد (طولشان به طول سیگنال بستگی دارد). به همین
دلیل این کلاس علاوه بر ذخیره‌ی خام، آماره‌های خلاصه هم می‌سازد.
"""

from __future__ import annotations

import numpy as np

from features import (
    butterworth_bandpass,
    echo_periodicity_autocorr,
    local_power_ratio,
    local_std_ratio,
    local_skewness_shift,
    local_crest_factor_ratio,
    local_spectral_centroid_ratio,
    local_dominant_frequency_ratio,
    local_entropy_ratio,
)

_ARRAY_KEYS = (
    "power_ratio",
    "std_ratio",
    "skewness_shift",
    "crest_factor_ratio",
    "spectral_centroid_ratio",
    "dominant_frequency_ratio",
    "entropy_ratio",
)


class SignalFeatures:
    """
    Parameters
    ----------
    fs : float
        نرخ نمونه‌برداری سیگنال (Hz) — برای فیلتر و ویژگی‌های طیفی لازم است.
    accuracy : int
        اندازه‌ی بلوک محلی، بین 1 تا 40 (پارامتر مشترک توابع local_*).
    bandpass : (low, high) یا None
        اگر داده شود، قبل از استخراج ویژگی یک فیلتر Butterworth
        band-pass روی سیگنال اعمال می‌شود.
    bandpass_order : int
        مرتبه‌ی فیلتر باندپس.
    autocorr_kwargs : dict
        پارامترهای اضافی برای echo_periodicity_autocorr
        (مثل min_period, max_period, analysis_start_time و ...).
    """

    def __init__(
        self,
        fs: float,
        accuracy: int = 10,
        bandpass: tuple[float, float] | None = None,
        bandpass_order: int = 4,
        autocorr_kwargs: dict | None = None,
    ) -> None:
        self.fs = fs
        self.accuracy = accuracy
        self.bandpass = bandpass
        self.bandpass_order = bandpass_order
        self.autocorr_kwargs = autocorr_kwargs or {}

        self.raw: dict = {}
        self.summary: dict = {}

    # ------------------------------------------------------------------
    def compute(self, signal: np.ndarray, time: np.ndarray) -> "SignalFeatures":
        """تمام توابع ویژگی را روی signal اجرا و نتیجه را در خود ذخیره می‌کند."""
        signal = np.asarray(signal, dtype=float)
        time = np.asarray(time, dtype=float)

        if self.bandpass is not None:
            low, high = self.bandpass
            signal = butterworth_bandpass(
                signal, self.fs, low, high, order=self.bandpass_order
            )

        period, strength = echo_periodicity_autocorr(
            signal, time, **self.autocorr_kwargs
        )

        self.raw = {
            "echo_period": period,
            "echo_strength": strength,
            "power_ratio": local_power_ratio(signal, self.accuracy),
            "std_ratio": local_std_ratio(signal, self.accuracy),
            "skewness_shift": local_skewness_shift(signal, self.accuracy),
            "crest_factor_ratio": local_crest_factor_ratio(signal, self.accuracy),
            "spectral_centroid_ratio": local_spectral_centroid_ratio(
                signal, self.fs, self.accuracy
            ),
            "dominant_frequency_ratio": local_dominant_frequency_ratio(
                signal, self.fs, self.accuracy
            ),
            "entropy_ratio": local_entropy_ratio(signal, self.accuracy),
        }

        self.summary = self._summarize(self.raw)
        return self

    # ------------------------------------------------------------------
    @staticmethod
    def _stats(arr: np.ndarray) -> dict:
        arr = np.asarray(arr, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def _summarize(self, raw: dict) -> dict:
        summary = {
            "echo_period": float(raw["echo_period"]),
            "echo_strength": float(raw["echo_strength"]),
        }
        for key in _ARRAY_KEYS:
            for stat_name, value in self._stats(raw[key]).items():
                summary[f"{key}_{stat_name}"] = value
        return summary

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """خروجی خلاصه به‌صورت دیکشنری (برای ساخت یک ردیف جدول)."""
        return dict(self.summary)

    def feature_names(self) -> list[str]:
        return list(self.summary.keys())

    def to_vector(self) -> np.ndarray:
        return np.array(list(self.summary.values()), dtype=float)
