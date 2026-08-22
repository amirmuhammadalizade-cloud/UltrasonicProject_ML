"""
train_model.py
================
1) دریافت دیتابیس از فایل (با کلاس File در loading.py)
2) ساخت جدول ویژگی برای هر سیگنال با کلاس SignalFeatures
3) برچسب‌گذاری خودکار هر سیگنال با «ضخامت واقعی» بر اساس موقعیت X آن،
   دقیقاً با همان منطق STEP_THICKNESS_MM / STEP_X_RANGE_MM که در
   Thickness_feature.py برای دیتاست Pk050 استفاده شده بود
4) آموزش و ارزیابی یک RandomForestRegressor روی این جدول

چرا Regressor و نه Classifier؟
--------------------------------
در Thickness_feature.py برچسب هر نمونه "true_thickness_mm" است — یک
عدد پیوسته (571.3, 452.0, 330.9, 210.8 میلی‌متر)، نه یک دسته‌ی گسسته.
پس این یک مسئله‌ی رگرسیون است و از RandomForestRegressor استفاده شده،
نه RandomForestClassifier.

چرا GroupKFold / Leave-One-Step-Out؟
--------------------------------------
همه‌ی نقاط y روی یک X ثابت، و همه‌ی X های داخل یک پله، عملاً از یک
نمونه‌ی فیزیکی با یک ضخامت واحد می‌آیند و به‌شدت به هم شبیه‌اند. یک
train_test_split تصادفی ساده باعث می‌شود داده از یک پله هم در آموزش
و هم در تست حاضر باشد (نشتِ داده / data leakage) و دقت مدل به‌طور
گمراه‌کننده‌ای بالا نشان داده شود. Thickness_feature.py دقیقاً به
همین دلیل از GroupKFold (با groups=step) و Leave-One-Step-Out استفاده
کرده؛ همان رویکرد اینجا هم پیاده شده است.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score

from loading import File
from features import not_zero_signal
from feature_extractor import SignalFeatures

# =========================================================================
# مشخصات پله‌های ضخامت دیتاست BAM Pk050 (همان مقادیر Thickness_feature.py)
# =========================================================================
STEP_THICKNESS_MM = {1: 571.3, 2: 452.0, 3: 330.9, 4: 210.8}
STEP_X_RANGE_MM = {
    1: (0, 500),
    2: (500, 1000),
    3: (1000, 1500),
    4: (1500, 2000),
}


def label_from_x(x_mm: float) -> tuple[float, int] | tuple[float, None]:
    """
    بر اساس موقعیت X (میلی‌متر)، پله و ضخامت واقعی متناظر را برمی‌گرداند.
    اگر X داخل هیچ‌کدام از بازه‌های شناخته‌شده نباشد، (nan, None) می‌دهد.
    """
    for step, (x_lo, x_hi) in STEP_X_RANGE_MM.items():
        if x_lo <= x_mm < x_hi:
            return STEP_THICKNESS_MM[step], step
    return float("nan"), None


def load_database() -> File:
    """
    داده‌های آلتراسونیک را با استفاده از کلاس File بارگذاری می‌کند.

    خروجی یک شیء File است که شامل:
      - .data : آرایه‌ی سه‌بعدی سیگنال‌ها، شکل (n_x, n_y, n_t)  (mmap شده)
      - .x, .y : مختصات مکانی
      - .t     : محور زمان هر سیگنال
    """
    return File()


def build_feature_table(
    db: File,
    fs: float,
    accuracy: int = 10,
    bandpass: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """
    برای هر سیگنال معتبر (غیرصفر) در دیتاست، یک بردار ویژگی می‌سازد،
    برچسب ضخامت واقعی و شماره‌ی پله را از روی X تعیین می‌کند، و همه را
    در یک DataFrame جمع می‌کند (هر ردیف = یک سیگنال).

    نقاطی که X آن‌ها داخل هیچ پله‌ای نیست (label نامعتبر) حذف می‌شوند.
    """
    data = db.data
    time = db.t

    valid_mask = not_zero_signal(data)  # شکل: (n_x, n_y)

    extractor = SignalFeatures(fs=fs, accuracy=accuracy, bandpass=bandpass)

    rows = []
    for i, j in zip(*np.nonzero(valid_mask)):
        x_mm = float(db.x[i])
        thickness, step = label_from_x(x_mm)
        if step is None:
            continue  # این نقطه داخل هیچ پله‌ی شناخته‌شده‌ای نیست

        signal = np.asarray(data[i, j, :])
        extractor.compute(signal, time)

        row = extractor.to_dict()
        row["x"] = x_mm
        row["y"] = float(db.y[j])
        row["step"] = step
        row["true_thickness_mm"] = thickness
        rows.append(row)

    table = pd.DataFrame(rows)
    print(f"تعداد نمونه‌های معتبر و برچسب‌دار: {len(table)}")
    if len(table):
        print(table.groupby("step")["true_thickness_mm"].agg(["count", "mean"]))
    return table


def evaluate_random_forest(
    table: pd.DataFrame,
    label_col: str = "true_thickness_mm",
    group_col: str = "step",
    drop_cols: tuple[str, ...] = ("x", "y"),
    random_state: int = 42,
    **rf_kwargs,
):
    """
    RandomForestRegressor را با GroupKFold (گروه‌بندی‌شده روی پله)
    ارزیابی می‌کند تا نشت داده بین پله‌ها رخ ندهد — دقیقاً هم‌ارز
    Leave-One-Step-Out در Thickness_feature.py وقتی تعداد fold برابر
    تعداد پله‌ها باشد.
    """
    feature_cols = [
        c for c in table.columns if c not in (label_col, group_col, *drop_cols)
    ]

    X = table[feature_cols].values
    y = table[label_col].values
    groups = table[group_col].values

    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=n_groups)

    rf_params = dict(n_estimators=300, random_state=random_state, n_jobs=-1)
    rf_params.update(rf_kwargs)
    model = RandomForestRegressor(**rf_params)

    preds = cross_val_predict(model, X, y, cv=gkf, groups=groups)

    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"\nGroupKFold (Leave-One-Step-Out) روی پله‌ها: MAE={mae:.2f} mm, R2={r2:.3f}")

    print("\nبه‌تفکیک پله (وقتی آن پله خارج از آموزش بوده):")
    for step in sorted(np.unique(groups)):
        mask = groups == step
        mae_step = mean_absolute_error(y[mask], preds[mask])
        print(
            f"  پله {step} (واقعی={STEP_THICKNESS_MM.get(step)} mm): "
            f"MAE={mae_step:.2f} mm, میانگین پیش‌بینی={preds[mask].mean():.1f} mm"
        )

    # مدل نهایی را روی کل داده فیت می‌کنیم تا برای استفاده‌ی عملی/استخراج
    # اهمیت ویژگی‌ها آماده باشد (این مدل دیگر برای ارزیابی دقت استفاده نشود،
    # چون روی همان داده‌ای فیت شده که MAE بالا از آن به‌دست آمد).
    model.fit(X, y)
    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("\nاهمیت ویژگی‌ها:\n", importances)

    return model, preds, importances


if __name__ == "__main__":
    FS = 2e6           # SAMPLING_RATE_HZ در Thickness_feature.py: 2 MS/s
    ACCURACY = 10       # پارامتر accuracy توابع local_*

    db = load_database()
    table = build_feature_table(db, fs=FS, accuracy=ACCURACY)

    if len(table) == 0:
        print("هیچ نمونه‌ی معتبری ساخته نشد — بازه‌های X یا مسیر داده را بررسی کنید.")
    else:
        model, preds, importances = evaluate_random_forest(table)
