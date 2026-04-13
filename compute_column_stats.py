import argparse
import warnings
warnings.filterwarnings("ignore")
 
import pandas as pd
import numpy as np
from scipy import stats

def compute_column_stats(
    series: pd.Series,
    target: pd.Series | None = None,
    target_pos: str | int = "yes",
    sentinel: float = 999,
    zscore_thr: float = 3.0,
) -> dict:
    """
    Обчислює повний набір статистик для однієї числової колонки.
 
    Параметри
    ----------
    series      : числова колонка датафрейму
    target      : цільова колонка (необов'язково)
    target_pos  : значення позитивного класу в target
    sentinel    : підозріле кодове значення (перевіряємо його частку)
    zscore_thr  : поріг для Z-score
 
    Повертає
    --------
    dict зі всіма метриками
    """
    s = series.dropna()
    n = len(s)
 
    # Базова статистика
    q1, q3   = s.quantile([0.25, 0.75])
    iqr      = q3 - q1
    lo, hi   = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_iqr    = int(((s < lo) | (s > hi)).sum())
    n_zscore = int((np.abs(stats.zscore(s)) > zscore_thr).sum())
 
    # Форма розподілу
    skewness = round(s.skew(), 4)
    kurt     = round(s.kurtosis(), 4)
 
    # Нормальність (Shapiro-Wilk на підвибірці)
    sample   = s.sample(min(n, 5000), random_state=42)
    _, p_sw  = stats.shapiro(sample)
 
    # Спеціальні значення
    n_zeros    = int((s == 0).sum())
    n_sentinel = int((s == sentinel).sum())
 
    # Унікальні значення
    n_unique = s.nunique()
    top5     = s.value_counts().head(5).to_dict()
 
    result = {
        "column"          : series.name,
        "n_total"         : n,
        "n_missing"       : int(series.isna().sum()),
        "min"             : round(float(s.min()), 4),
        "max"             : round(float(s.max()), 4),
        "mean"            : round(float(s.mean()), 4),
        "median"          : round(float(s.median()), 4),
        "std"             : round(float(s.std()), 4),
        "q1"              : round(float(q1), 4),
        "q3"              : round(float(q3), 4),
        "iqr"             : round(float(iqr), 4),
        "iqr_lower_fence" : round(float(lo), 4),
        "iqr_upper_fence" : round(float(hi), 4),
        "n_iqr_outliers"  : n_iqr,
        "pct_iqr_outliers": round(n_iqr / n * 100, 2),
        "n_zscore_outliers": n_zscore,
        "pct_zscore_outliers": round(n_zscore / n * 100, 2),
        "skewness"        : skewness,
        "kurtosis"        : kurt,
        "shapiro_p"       : round(float(p_sw), 6),
        "is_normal"       : p_sw > 0.05,
        "n_zeros"         : n_zeros,
        "pct_zeros"       : round(n_zeros / n * 100, 2),
        f"n_sentinel_{int(sentinel)}": n_sentinel,
        f"pct_sentinel_{int(sentinel)}": round(n_sentinel / n * 100, 2),
        "n_unique"        : n_unique,
        "top5_values"     : str(top5),
    }

    return result

