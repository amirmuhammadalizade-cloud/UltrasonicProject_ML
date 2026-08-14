"""
depth_estimation.py

ماژول برآورد عمق برای پروژه تحلیل عمق دیوار بتنی با روش التراسونیک (pulse-echo).

این ماژول سه بخش اصلی را پیاده‌سازی می‌کند:
    - DepthCalibration: کالیبراسیون خطی تجربی بین زمان echo دیواره پشتی و
      ضخامت واقعی؛ خروجی آن یک *ویژگی* برای مدل یادگیری ماشین است.
    - GroundTruthDepth: عمق واقعی بر اساس هندسه شناخته‌شده نمونه؛ خروجی آن
      *برچسب* (label) برای آموزش مدل یادگیری ماشین است.
    - DatasetBuilder: اجرای کامل pipeline استخراج ویژگی روی تمام نقاط معتبر
      شبکه اندازه‌گیری و ساخت ردیف‌های دیتاست نهایی.

نکتهٔ معماری: عمق واقعی (label) همیشه باید از هندسهٔ فیزیکی مستقل نمونه
گرفته شود، نه از کالیبراسیون زمان→عمق. اگر از خروجی کالیبراسیون به‌عنوان
برچسب استفاده شود، مدل صرفاً یاد می‌گیرد همان تخمین را بازتولید کند
(استدلال دوری/circular reasoning) و دقتش هرگز از دقت خود کالیبراسیون
فراتر نمی‌رود. به همین دلیل کالیبراسیون فقط به‌عنوان یک feature ورودی
مکمل (baseline تحلیلی که مدل روی آن بهبود می‌دهد) استفاده می‌شود.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from signal_processing import FeatureExtractor, Signal, SignalFeatures


def signal_to_noise_ratio(signal: np.ndarray, noise_window: slice) -> float:
    """
    نسبت سیگنال به نویز (SNR) یک A-scan را بر حسب دسی‌بل محاسبه می‌کند.

    SNR به‌صورت نسبت توان کل سیگنال به توان نویز در بازه‌ای ابتدایی که
    فاقد echo است (noise_window) تعریف می‌شود.

    Args:
        signal: آرایه یک‌بعدی دامنه A-scan در طول زمان.
        noise_window: برشی (slice) از ابتدای سیگنال که فقط نویز را در بر
            می‌گیرد و به‌عنوان مرجع محاسبه توان نویز استفاده می‌شود.

    Returns:
        مقدار SNR بر حسب دسی‌بل (dB). اگر توان نویز صفر باشد، برای
        جلوگیری از تقسیم بر صفر، ``float("inf")`` بازگردانده می‌شود.
    """
    noise_segment = signal[noise_window]
    noise_power = float(np.mean(np.square(noise_segment)))
    signal_power = float(np.mean(np.square(signal)))

    if noise_power <= 0.0:
        return float("inf")

    return float(10.0 * np.log10(signal_power / noise_power))


@dataclass
class DepthCalibration:
    """
    کالیبراسیون خطی تجربی بین زمان پرواز echo دیواره پشتی (time of flight)
    و ضخامت واقعی نمونه.

    این کالیبراسیون جایگزین مدل فیزیکی ساده ``d = v * t / 2`` شده است، چون
    هم دقت تجربی بالاتری دارد و هم افست تأخیر سیستم (system delay offset)
    را به‌طور ضمنی پوشش می‌دهد.

    توجه: خروجی این کلاس فقط باید به‌عنوان یک ویژگی ورودی (feature) برای
    مدل یادگیری ماشین استفاده شود، نه به‌عنوان برچسب واقعی (label)؛ عمق
    واقعی باید از هندسهٔ مستقل نمونه گرفته شود (به ``GroundTruthDepth``
    مراجعه کنید).

    Attributes:
        slope: شیب خط برازش‌شده (واحد: میکروثانیه بر میلی‌متر، µs/mm).
        intercept: عرض از مبدأ خط برازش‌شده (واحد: میکروثانیه، µs).
    """

    slope: float
    intercept: float

    @classmethod
    def fit(
        cls,
        known_thickness_mm: np.ndarray,
        measured_time_us: np.ndarray,
    ) -> "DepthCalibration":
        """
        یک کالیبراسیون خطی درجه یک بین ضخامت شناخته‌شده و زمان echo
        اندازه‌گیری‌شدهٔ متناظر برازش می‌کند.

        رابطه برازش‌شده به فرم زیر است::

            measured_time_us ≈ slope * known_thickness_mm + intercept

        Args:
            known_thickness_mm: آرایه‌ای از ضخامت‌های واقعی و شناخته‌شدهٔ
                نمونه (بر حسب میلی‌متر)، برگرفته از هندسهٔ نمونه.
            measured_time_us: آرایه‌ای هم‌طول از زمان echo دیواره پشتی
                اندازه‌گیری‌شده در همان نقاط (بر حسب میکروثانیه).

        Returns:
            نمونه‌ای از ``DepthCalibration`` با ``slope`` و ``intercept``
            برازش‌شده.
        """
        slope, intercept = np.polyfit(known_thickness_mm, measured_time_us, 1)
        return cls(slope=float(slope), intercept=float(intercept))

    def estimate_depth(self, time_of_flight: float) -> float:
        """
        رابطه خطی کالیبراسیون را معکوس می‌کند تا عمق را از روی زمان پرواز
        echo تخمین بزند.

        رابطه معکوس‌شده::

            depth = (time_of_flight - intercept) / slope

        Args:
            time_of_flight: زمان پرواز echo دیواره پشتی (بر حسب
                میکروثانیه).

        Returns:
            عمق تخمینی بر حسب میلی‌متر (صرفاً به‌عنوان یک feature، نه
            برچسب واقعی).
        """
        return (time_of_flight - self.intercept) / self.slope


@dataclass
class GroundTruthDepth:
    """
    عمق واقعی (ground truth) بر اساس هندسه شناخته‌شده و مستقل از سیگنال
    نمونه پلکانی.

    این کلاس منبع برچسب (label) برای آموزش مدل یادگیری ماشین است، زیرا
    مقدار آن مستقیماً از طراحی/ساخت فیزیکی نمونه می‌آید و خطای اندازه‌گیری
    صفر دارد — برخلاف ``DepthCalibration`` که یک تخمین مبتنی بر سیگنال
    است.

    Attributes:
        step_boundaries_mm: مرزهای موقعیت X ابتدای هر پله بر حسب میلی‌متر،
            به‌ترتیب صعودی (مثلاً ``[0, 500, 1000, 1500]``).
        step_thicknesses_mm: ضخامت واقعی هر پله بر حسب میلی‌متر، به همان
            ترتیب پله‌ها (مثلاً ``[571.3, 452.0, 330.9, 210.8]``).
    """

    step_boundaries_mm: np.ndarray
    step_thicknesses_mm: np.ndarray

    def __post_init__(self) -> None:
        self.step_boundaries_mm = np.asarray(self.step_boundaries_mm, dtype=float)
        self.step_thicknesses_mm = np.asarray(self.step_thicknesses_mm, dtype=float)

    def depth_at(self, x_position_mm: float) -> float:
        """
        ضخامت واقعی نمونه را در یک موقعیت مشخص روی محور X برمی‌گرداند.

        با ``np.searchsorted`` مشخص می‌شود موقعیت داده‌شده در کدام بازهٔ
        (پلهٔ) ``step_boundaries_mm`` قرار می‌گیرد و ضخامت همان پله از
        ``step_thicknesses_mm`` بازگردانده می‌شود.

        Args:
            x_position_mm: موقعیت روی محور X (طول نمونه) بر حسب میلی‌متر.

        Returns:
            ضخامت واقعی پلهٔ متناظر با آن موقعیت، بر حسب میلی‌متر.
        """
        step_index = (
            np.searchsorted(self.step_boundaries_mm, x_position_mm, side="right") - 1
        )
        step_index = int(np.clip(step_index, 0, len(self.step_thicknesses_mm) - 1))
        return float(self.step_thicknesses_mm[step_index])


class DatasetBuilder:
    """
    اجرای کامل pipeline استخراج ویژگی روی تمام نقاط معتبر شبکهٔ اندازه‌گیری
    و ساخت دیتاست نهایی برای آموزش مدل یادگیری ماشین.

    این کلاس منطق موجود در ``Signal`` و ``FeatureExtractor`` را با
    ``DepthCalibration`` (feature) و ``GroundTruthDepth`` (label) ترکیب
    می‌کند؛ هیچ منطق پردازش سیگنالی را دوباره پیاده‌سازی نمی‌کند.
    """

    def __init__(
        self,
        signal_obj: Signal,
        depth_calibration: DepthCalibration,
        ground_truth: GroundTruthDepth,
        filter_params: dict,
        noise_window: slice,
    ) -> None:
        """
        Args:
            signal_obj: نمونهٔ ``Signal`` حاوی دادهٔ خام شبکهٔ اندازه‌گیری
                (data, x, y, z).
            depth_calibration: کالیبراسیون زمان→عمق برازش‌شده (برای feature
                ``calibrated_depth_estimate``).
            ground_truth: تولیدکنندهٔ عمق واقعی بر اساس هندسهٔ نمونه (برای
                label ``true_depth``).
            filter_params: دیکشنری حاوی کلید ``steps`` که مستقیماً به
                ``Signal.set_filter`` پاس داده می‌شود، مثلاً:
                ``{"steps": [("gaussian", {"sigma": 3})]}``. با این فرمت
                کاربر می‌تواند هر ترکیب و ترتیب دلخواهی از فیلترها
                (gaussian, median, uniform, savgol) را مشخص کند.
            noise_window: برش (slice) ابتدای هر A-scan که فاقد echo است و
                برای محاسبهٔ SNR استفاده می‌شود.
        """
        self.signal_obj = signal_obj
        self.depth_calibration = depth_calibration
        self.ground_truth = ground_truth
        self.filter_params = filter_params
        self.noise_window = noise_window

    def build(self) -> List[Dict]:
        """
        pipeline را روی تمام نقاط معتبر شبکه (بر اساس
        ``non_empty_value_mask``) اجرا می‌کند و لیستی از ردیف‌های دیتاست
        را برمی‌گرداند.

        برای هر نقطهٔ معتبر ``(x_index, y_index)``:
            1. envelope خام محاسبه و طبق ``filter_params`` فیلتر می‌شود.
            2. ``SignalFeatures`` با ``FeatureExtractor`` استخراج می‌شود
               (شامل ``time_of_flight``).
            3. ``snr`` با ``signal_to_noise_ratio`` محاسبه می‌شود.
            4. ``calibrated_depth_estimate`` (feature) از
               ``DepthCalibration.estimate_depth`` محاسبه می‌شود.
            5. ``true_depth`` (label) از ``GroundTruthDepth.depth_at``
               محاسبه می‌شود.

        Returns:
            لیستی از دیکشنری‌ها؛ هر دیکشنری یک ردیف دیتاست است و مستقیماً
            قابل تبدیل به ``pandas.DataFrame`` است (مثلاً
            ``pd.DataFrame(rows)``).
        """
        rows: List[Dict] = []

        mask = self.signal_obj.non_empty_value_mask()
        time_axis = self.signal_obj.Time()
        time_step_us = float(time_axis[1] - time_axis[0])
        sampling_rate_hz = 1.0 / (time_step_us * 1e-6)

        x_indices, y_indices = np.nonzero(mask)

        for x_index, y_index in zip(x_indices, y_indices):
            x_index = int(x_index)
            y_index = int(y_index)

            raw_envelope = self.signal_obj.compute_envelope(x_index, y_index)
            filtered_signal = self.signal_obj.set_filter(
                raw_envelope, **self.filter_params
            ).apply()

            # فرض معماری: FeatureExtractor(signal, time, sampling_rate).extract()
            # یک SignalFeatures برمی‌گرداند (طبق توضیح داده‌شده در
            # signal_processing.py؛ اگر نام متد در پیاده‌سازی واقعی شما
            # متفاوت است، فقط همین خط را اصلاح کنید).
            features: SignalFeatures = FeatureExtractor(
                filtered_signal, time_axis, sampling_rate_hz
            ).extract()

            snr = signal_to_noise_ratio(filtered_signal, self.noise_window)

            calibrated_depth_estimate = self.depth_calibration.estimate_depth(
                features.time_of_flight
            )

            x_position_mm = float(self.signal_obj.x[x_index])
            y_position_mm = float(self.signal_obj.y[y_index])
            true_depth = self.ground_truth.depth_at(x_position_mm)

            row: Dict = {
                "x_index": x_index,
                "y_index": y_index,
                "x_position_mm": x_position_mm,
                "y_position_mm": y_position_mm,
                **features.to_dict(),
                "snr": snr,
                "calibrated_depth_estimate": calibrated_depth_estimate,
                "true_depth": true_depth,
            }
            rows.append(row)

        return rows


if __name__ == "__main__":
    # --- مثال کوتاه استفاده ---

    # 1) کالیبراسیون تجربی زمان → عمق (از داده‌های چهار نقطهٔ شناخته‌شده)
    calibration = DepthCalibration.fit(
        known_thickness_mm=np.array([571.3, 452.0, 330.9, 210.8]),
        measured_time_us=np.array([273.0, 230.5, 170.0, 113.0]),
    )
    print(f"calibration: slope={calibration.slope:.4f} µs/mm, "
          f"intercept={calibration.intercept:.4f} µs")

    # 2) عمق واقعی بر اساس هندسهٔ نمونهٔ پلکانی Pk050
    ground_truth = GroundTruthDepth(
        step_boundaries_mm=np.array([0, 500, 1000, 1500]),
        step_thicknesses_mm=np.array([571.3, 452.0, 330.9, 210.8]),
    )

    # 3) بارگذاری Signal واقعی از دیتاست BAM (اینجا فقط جایگزین نمایشی):
    from data_loader import load_sample
    raw_bam_data, x_axis_mm, y_axis_mm, z_axis = load_sample('Pk050_3D_Dataset_Long_Rot00')
    signal_obj = Signal(data=raw_bam_data, x=x_axis_mm, y=y_axis_mm, z=z_axis)

    print(signal_obj.data.shape)
    print(signal_obj.non_empty_value_mask().sum())  # تعداد نقاط معتبر
    
    builder = DatasetBuilder(
         signal_obj=signal_obj,
         depth_calibration=calibration,
         ground_truth=ground_truth,
         filter_params={"steps": [("gaussian", {"sigma": 3})]},
         noise_window=slice(0, 40),
     )
    dataset_rows = builder.build()
    
    import pandas as pd
    df = pd.DataFrame(dataset_rows)
    print(df.head())