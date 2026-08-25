"""
main.py
=======
اجرای اصلیِ پروژه: پیمایشِ دیتاست، نمونه‌گیریِ تصادفی از هر پله‌ی ضخامت،
محاسبه‌ی فیچرها برای هر A-scan، و درنهایت اجرای رگرسیون/تشخیص/آمار روی
نتیجه.

این فایل معادل بخش ۸ اسکریپت اصلی است (اجرای اصلی — main).

بقیه‌ی منطقِ پروژه به‌ترتیب در فایل‌های زیر قرار دارد و این فایل فقط
آن‌ها را کنارِ هم می‌چیند:
    - config.py       : بارگذاریِ داده + ثابت‌های هندسه‌ی نمونه/فیچرها
    - signal_utils.py  : توابعِ کمکیِ سطحِ پایینِ سیگنال
    - features.py       : استخراجِ فیچر از هر A-scan (compute_all_features)
    - modeling.py       : اعتبارسنجیِ دیتاست و رگرسیون/تشخیص/baseline

برای اجرا کافی است این فایل را مستقیم اجرا کنی:
    python main.py
"""

import numpy as np

from config import (
    data,
    x_coords,
    y_coords,
    time_us,
    SAMPLING_RATE_HZ,
    STEP_THICKNESS_MM,
    STEP_X_RANGE_MM,
    N_SAMPLES_PER_STEP,
    RANDOM_SEED,
    REQUIRED_FINITE_KEYS,
)
from signal_utils import is_zero_signal
from features import compute_all_features
from modeling import (
    validate_dataset,
    run_regression_models,
    run_diagnostics,
    run_linear_baseline,
    print_feature_stats_per_step,
)


def main():
    """
    جریانِ کاملِ پردازش:
        1) اعتبارسنجیِ دیتاست (validate_dataset) و چاپِ اطلاعاتِ کلیِ آن
        2) برای هر پله‌ی ضخامت: انتخابِ تصادفیِ حداکثر N_SAMPLES_PER_STEP
           موقعیتِ X (با seedِ ثابتِ RANDOM_SEED برای تکرارپذیری)، سپس
           پیمایشِ همه‌ی موقعیت‌های Y در همان X و محاسبه‌ی فیچرِ هر A-scan
        3) دورریختنِ نمونه‌های نامعتبر با سه دسته‌ی شکست که جداگانه شمارش
           می‌شوند:
               - zero_signal      : سیگنالِ کاملاً صفر (نقطه‌ی padding،
                                     خارج از سطحِ واقعیِ نمونه)
               - ValueError       : یکی از توابعِ فیچر (مثلاً
                                     echo_periodicity_autocorr) نتوانسته
                                     تناوبِ معتبری تخمین بزند
               - invalid_feature  : فیچرها محاسبه شدند اما یکی از
                                     REQUIRED_FINITE_KEYS مقدارِ NaN/Inf
                                     دارد
        4) چاپِ گزارشِ خلاصه به تفکیکِ پله، سپس اجرایِ آمارِ فیچرها
           (print_feature_stats_per_step)، رگرسیونِ Random Forest
           (run_regression_models)، تشخیص‌های عمیق‌تر (run_diagnostics)،
           و دو baseline خطی (run_linear_baseline) روی echo_period_us و
           echo_period_linfit

    خروجی:
        ندارد — همه‌ی نتایج در طولِ اجرا چاپ می‌شوند و نمودارِ تشخیصی به
        فایلِ PNG ذخیره می‌شود.
    """
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

    run_regression_models(rows)

    run_diagnostics(rows)
    run_linear_baseline(rows, feature_name="echo_period_us")
    run_linear_baseline(rows, feature_name="echo_period_linfit")


if __name__ == "__main__":
    main()
