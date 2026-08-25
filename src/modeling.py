"""
modeling.py
===========
اعتبارسنجیِ دیتاست، و توابعِ رگرسیون/ارزیابیِ مدل برای پیش‌بینیِ ضخامتِ
واقعی از روی فیچرهایی که در features.py استخراج شده‌اند.

این فایل معادل بخش‌های ۶ و ۷ اسکریپت اصلی است:
  6. اعتبارسنجی دیتاست
  7. رگرسیون
     7.5 تشخیص و ارزیابیِ عمیق‌ترِ مدل (نمودار پراکندگی + خطا به تفکیکِ
         پله + Leave-One-Step-Out)
     7.6 baseline خطی (برای مقایسه با Random Forest در تعمیم به ضخامتِ ندیده)
     7.7 آمارِ فیچرهای کلیدی به تفکیکِ پله + سرعتِ ضمنیِ موج

این فایل از FEATURE_NAMES و STEP_THICKNESS_MM در config.py استفاده می‌کند؛
خودِ فیچرها را محاسبه نمی‌کند (آن‌ها از قبل در rows حاضرند و در
main.py توسط compute_all_features از features.py ساخته می‌شوند).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

from config import FEATURE_NAMES, STEP_THICKNESS_MM


# =========================================================================
# 6. اعتبارسنجی دیتاست
# =========================================================================
def validate_dataset(data, x_coords, y_coords, time_us, sampling_rate_hz):
    """
    قبل از شروعِ پردازش، ابعادِ داده و نرخِ نمونه‌برداری را بررسی می‌کند.
    اگر چیزی ناسازگار باشد، به‌جای یک خطای گنگ در وسطِ پردازشِ بعدی، همین‌جا
    یک خطای واضح و قابلِ‌فهم صادر می‌شود.

    پارامترها:
        data              : آرایه‌ی سه‌بعدیِ (X, Y, time) دامنه‌ی سیگنال
        x_coords          : مختصاتِ محورِ X (باید هم‌طول با data.shape[0] باشد)
        y_coords          : مختصاتِ محورِ Y (باید هم‌طول با data.shape[1] باشد)
        time_us           : محورِ زمان بر حسبِ میکروثانیه
                             (باید هم‌طول با data.shape[2] باشد)
        sampling_rate_hz  : نرخِ نمونه‌برداریِ موردِ انتظار (هرتز)، برای
                             مقایسه با فاصله‌ی زمانیِ واقعاً اندازه‌گیری‌شده
                             در time_us

    خروجی:
        dt_measured : فاصله‌ی زمانیِ واقعاً اندازه‌گیری‌شده در time_us
                      (میکروثانیه)

    خطا:
        اگر ابعادِ آرایه‌ها با هم مطابقت نداشته باشند، یا فاصله‌ی زمانیِ
        اندازه‌گیری‌شده با نرخِ نمونه‌برداریِ موردِ انتظار همخوانی نداشته
        باشد (بیش از ۵٪ اختلاف)، ValueError صادر می‌کند.
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
    """
    مدلِ Random Forest را با جست‌وجوی هایپرپارامتر (RandomizedSearchCV) روی
    GroupKFold (گروه‌بندی‌شده بر اساسِ موقعیتِ X، تا نشتِ داده بینِ train/test
    از یک موقعیتِ فیزیکیِ یکسان رخ ندهد) تنظیم و ارزیابی می‌کند. در انتها
    اهمیتِ هر فیچر را هم چاپ می‌کند.

    پارامترها:
        rows : لیستی از دیکشنری‌ها؛ هر دیکشنری باید کلیدهای FEATURE_NAMES،
               "true_thickness_mm" و "x_idx" را داشته باشد (خروجیِ main()
               در main.py)

    خروجی:
        ندارد — نتایج (بهترین هایپرپارامترها، MAE/RMSE/R2، اهمیتِ فیچرها)
        مستقیماً چاپ می‌شوند.
    """
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

    # ارزیابیِ نهایی با cross_val_predict روی همان تقسیم‌بندیِ GroupKFold،
    # با بهترین هایپرپارامترها (تا معیارِ MAE/RMSE/R2 مستقل از foldِ
    # انتخاب‌شده در جست‌وجو باشد)
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
    سه تشخیصِ کلیدی را روی rows انجام می‌دهد:
        ۱) نمودارِ پراکندگیِ echo_period_us و echo_period_linfit در برابرِ
           ضخامتِ واقعی (و ذخیره‌ی آن به‌عنوانِ PNG)
        ۲) خطای مدل به تفکیکِ هر پله (روی همان GroupKFold مبتنی بر موقعیتِ X)
        ۳) ارزیابیِ Leave-One-Step-Out: آموزش روی ۳ پله و تست روی پله‌ی
           چهارمی که مدل هرگز ندیده — معیارِ واقعی‌تری از تعمیم‌پذیری به
           ضخامتِ کاملاً جدید، نسبت به GroupKFold روی X در همان ۴ پله.

    پارامترها:
        rows       : لیستی از دیکشنری‌های فیچر (مثلِ run_regression_models)؛
                     علاوه‌بر FEATURE_NAMES و true_thickness_mm و x_idx، باید
                     کلیدِ "step" (شماره‌ی پله) را هم داشته باشد.
        output_dir : پوشه‌ای که نمودارِ پراکندگی در آن ذخیره می‌شود. اگر
                     مشخص نشود، پوشه‌ی جاری (".") استفاده می‌شود.

    خروجی:
        ندارد — نمودار به‌صورتِ فایلِ PNG ذخیره می‌شود و بقیه‌ی نتایج چاپ
        می‌شوند.
    """
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
    یک رگرسیونِ خطیِ ساده روی یک فیچر (پیش‌فرض echo_period_us) با ضخامتِ
    واقعی برازش می‌کند و همان ارزیابیِ GroupKFold + Leave-One-Step-Out را
    روی آن اجرا می‌کند. هدف: بررسیِ این‌که آیا مدلِ خطی — که برخلافِ RF
    می‌تواند خارج از بازه‌ی دیده‌شده هم extrapolate کند — به ضخامت‌های
    کاملاً جدید بهتر تعمیم می‌دهد یا نه.

    پارامترها:
        rows         : لیستی از دیکشنری‌های فیچر (مثلِ run_diagnostics)
        feature_name : نامِ فیچری که رگرسیونِ خطی روی آن اجرا می‌شود؛ باید
                       عضوی از FEATURE_NAMES باشد.

    خروجی:
        ندارد — نتایج (MAE/R2 برایِ GroupKFold و MAE برایِ هر پله در
        Leave-One-Step-Out) مستقیماً چاپ می‌شوند.
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
    برای هر پله، میانگین/انحراف‌معیار/میانه‌ی فیچرهای کلیدی و سرعتِ موجِ
    ضمنی (implied velocity = 2*d/period) را چاپ می‌کند. اگر سرعتِ ضمنی
    بینِ پله‌ها ناسازگار باشد یا پراکندگیِ زیادی داشته باشد، یعنی خودِ
    تخمینِ period به‌اندازه‌ی کافی دقیق/پایدار نیست.

    پارامترها:
        rows          : لیستی از دیکشنری‌های فیچر که حداقل کلیدِ "step" و
                         هر یک از feature_names را دارند.
        feature_names : لیستِ نامِ فیچرهایی که آمارشان چاپ می‌شود. اگر
                         مشخص نشود، پیش‌فرض
                         ["echo_period_us", "echo_period_linfit",
                          "echo_spacing_precise"] استفاده می‌شود.

    خروجی:
        ندارد — آمار مستقیماً چاپ می‌شود.
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
