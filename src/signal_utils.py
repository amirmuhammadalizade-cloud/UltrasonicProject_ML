"""
signal_utils.py
===============
توابع کمکی سطح پایین که مستقیماً روی سیگنال A-scan یا پنجره‌های زمانی آن
کار می‌کنند، اما به‌تنهایی یک «فیچر» تولید نمی‌کنند — بلکه ابزارهایی
هستند که توابع استخراج فیچر (در features.py) از آن‌ها استفاده می‌کنند.

این فایل معادل بخش ۳ اسکریپت اصلی است. علاوه بر آن، دو تابع کمکی
`_parabolic_peak` و `get_backwall_echoes` هم اینجا آمده‌اند؛ در فایل
اصلی این دو در کنار توابع فیچر (بخش ۴) نوشته شده بودند، اما چون خودشان
مستقیماً یک عدد فیچر تولید نمی‌کنند (فقط پنجره/پیک استخراج می‌کنند)،
منطقاً محل مناسب‌ترشان همین‌جا، کنار بقیه‌ی ابزارهای سیگنال است.
"""

import numpy as np


def extract_window(signal, time_arr, center_time, half_window):
    """
    یک پنجره‌ی زمانی به مرکز و نیم‌عرض مشخص از سیگنال استخراج می‌کند.

    پارامترها:
        signal      : آرایه‌ی دامنه‌ی سیگنال
        time_arr    : آرایه‌ی زمان متناظر با signal (باید هم‌طول باشند)
        center_time : مرکز پنجره‌ی موردنظر (همان واحد زمانی time_arr)
        half_window : نیم‌عرض پنجره — پنجره از
                      (center_time - half_window) تا (center_time + half_window)
                      را در بر می‌گیرد.

    خروجی:
        tuple به‌شکل (signal_in_window, time_in_window)
    """
    mask = (time_arr >= center_time - half_window) & (time_arr <= center_time + half_window)
    return signal[mask], time_arr[mask]


def is_zero_signal(signal):
    """
    تشخیص می‌دهد که آیا یک سیگنال کاملاً صفر است یا نه (یعنی نقطه‌ای
    خارج از سطح واقعی نمونه است که در دیتاست با صفر پر شده - padding).

    چرا از np.allclose استفاده نشده؟
    -------------------------------------------------------------------
    np.allclose ممکن است سیگنال‌های واقعی با دامنه‌ی خیلی کم (نویز ضعیف)
    را هم اشتباهاً «صفر» تشخیص دهد. به همین دلیل اینجا فقط حالتی را
    «صفر» در نظر می‌گیریم که تک‌تکِ مقادیر دقیقاً برابر صفر باشند؛ این
    دقیقاً همان الگویی است که برای padding استفاده می‌شود، در حالی که
    نویز واقعیِ اندازه‌گیری‌شده تقریباً هرگز دقیقاً صفر نیست.

    خروجی:
        True اگر تمام مقادیر سیگنال صفر باشند، در غیر این صورت False.
    """
    return not np.any(signal)


def _parabolic_peak(y, idx):
    """
    درون‌یابی سهمی (parabolic interpolation) اطراف یک پیکِ گسسته، برای
    تخمینِ دقیق‌ترِ موقعیت زیرنمونه‌ای (sub-sample) آن پیک.

    ایده‌ی کار:
        اگر پیک را فقط برابر با اندیس صحیح idx در نظر بگیریم، دقتِ
        تخمین محدود به فاصله‌ی نمونه‌برداری (dt) می‌شود. اما با فیت یک
        سهمی (پارابولا) روی سه نقطه‌ی مجاورِ پیک [idx-1, idx, idx+1]،
        می‌توان فهمید که رأسِ واقعیِ پیک چقدر (به‌صورت یک عدد اعشاری،
        معمولاً بین -0.5 و +0.5) از idx جابه‌جا شده است.

    پارامترها:
        y   : آرایه‌ی مقادیر (مثلاً دامنه‌ی سیگنال یا خروجی کراس‌کورلیشن)
        idx : اندیسِ پیکِ گسسته (که قبلاً با argmax پیدا شده)

    خروجی:
        مقدار جابه‌جاییِ زیرنمونه‌ای (sub-sample shift) به‌عنوان عدد
        اعشاری. اگر idx روی لبه‌ی آرایه باشد یا داده برای فیت کافی
        نباشد، مقدار 0.0 برگردانده می‌شود (یعنی بدون اصلاح).
    """
    if idx <= 0 or idx >= len(y) - 1:
        return 0.0
    y0, y1, y2 = y[idx - 1], y[idx], y[idx + 1]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return 0.0
    return 0.5 * (y0 - y2) / denom


def get_backwall_echoes(signal, time_arr, first_backwall, half_window=50,
                         max_echoes=4, search_margin=40):
    """
    اکوهای برگشتی متوالی (backwall echoes) را در سیگنال پیدا می‌کند، با
    شروع از تناوبِ تخمینیِ first_backwall (که معمولاً از
    echo_periodicity_autocorr در features.py به‌دست می‌آید).

    منطقِ کار:
        برای اکوی n-ام، مرکزِ تقریبی‌اش برابر با n * first_backwall است.
        دورِ این مرکز یک پنجره‌ی جست‌وجوی نسبتاً بزرگ‌تر
        (half_window + search_margin) باز می‌کنیم، بزرگ‌ترین دامنه در
        همان بازه را «پیکِ واقعیِ اکو» در نظر می‌گیریم، و سپس یک پنجره‌ی
        نهاییِ باریک‌تر (half_window) دقیقاً حول همان پیک استخراج
        می‌کنیم و برمی‌گردانیم.

    نکته‌ی مهم:
        این تابع فرض نمی‌کند که حتماً max_echoes اکو در سیگنال وجود
        دارد. به‌محض اینکه اکویی پیدا نشود یا خارج از بازه‌ی زمانیِ
        سیگنال باشد، حلقه متوقف می‌شود — یعنی همیشه فقط همان اکوهایی
        برگردانده می‌شوند که واقعاً قابل‌تشخیص بوده‌اند.

    پارامترها:
        signal, time_arr : سیگنال و محور زمانِ متناظر با آن
        first_backwall   : تناوبِ تخمینیِ بین اکوهای متوالی
        half_window      : نیم‌عرضِ پنجره‌ی نهایی برای هر اکو
        max_echoes       : حداکثر تعداد اکویی که تلاش می‌شود پیدا شود
        search_margin    : حاشیه‌ی اضافه در جست‌وجوی پیکِ واقعیِ هر اکو

    خروجی:
        دیکشنری به‌شکل {"echo_1": (t_win, sig_win), "echo_2": (...), ...}
        که هر مقدار، پنجره‌ی زمان و سیگنالِ متناظر با همان اکو است.

    خطا:
        اگر first_backwall نامعتبر (غیرمحدود یا غیرمثبت) باشد، ValueError
        صادر می‌کند.
    """
    if not np.isfinite(first_backwall) or first_backwall <= 0:
        raise ValueError("تناوب اکو (first_backwall) نامعتبر است.")

    echoes = {}
    t_max = time_arr[-1]
    for n in range(1, max_echoes + 1):
        approx_center = n * first_backwall
        if approx_center - (half_window + search_margin) > t_max:
            break  # این اکو و اکوهای بعدی خارج از بازه‌ی زمانی سیگنال‌اند

        search_sig, search_t = extract_window(
            signal, time_arr, approx_center, half_window + search_margin
        )
        if len(search_sig) == 0:
            break

        peak_idx = np.argmax(np.abs(search_sig))
        real_center = search_t[peak_idx]
        sig_win, t_win = extract_window(signal, time_arr, real_center, half_window)
        if len(t_win) == 0 or len(sig_win) == 0:
            break

        echoes[f"echo_{n}"] = (t_win, sig_win)
    return echoes