import numpy as np


def non_empty_mask(data: np.ndarray) -> np.ndarray:
    """ماسک نقاطی از شبکه اندازه‌گیری که داده واقعی دارند (غیر صفر)."""
    return np.any(data != 0, axis=2)


def crop_to_measured_region(
    data: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """بخش‌های صفر (اندازه‌گیری‌نشده) در حاشیه شبکه را حذف می‌کند.

    شبکه اندازه‌گیری اصلی (187x71) داخل یک شبکه مستطیلی بزرگ‌تر
    (201x81) قرار گرفته و نقاط خارج از محدوده واقعی با بردار صفر
    پر شده‌اند. این تابع کوچک‌ترین مستطیل شامل تمام نقاط غیر صفر را
    پیدا کرده و داده و مختصات x, y را به همان محدوده برش می‌زند.

    Args:
        data: آرایه سه‌بعدی خام (X, Y, time).
        x: مختصات محور X.
        y: مختصات محور Y.

    Returns:
        داده و مختصات برش‌خورده به همان ترتیب (data, x, y).
    """
    mask = non_empty_mask(data)

    x_valid = np.where(mask.any(axis=1))[0]
    y_valid = np.where(mask.any(axis=0))[0]

    if x_valid.size == 0 or y_valid.size == 0:
        raise ValueError("هیچ نقطه اندازه‌گیری‌شده‌ای در داده یافت نشد.")

    x_start, x_end = x_valid[0], x_valid[-1] + 1
    y_start, y_end = y_valid[0], y_valid[-1] + 1

    cropped_data = data[x_start:x_end, y_start:y_end, :]
    cropped_x = x[x_start:x_end]
    cropped_y = y[y_start:y_end]

    return cropped_data, cropped_x, cropped_y