"""
plot_conversion.py — Функція для візуалізації конверсії по категорії
=====================================================================
Використання:
    from plot_conversion import plot_conversion

    plot_conversion(df, col="job")
    plot_conversion(df, col="month", target_col="y", target_pos="yes")
    plot_conversion(df, col="month", order=["jan","feb","mar",...])
    plot_conversion(df, col="job", top_n=8, save_path="job_conv.png")
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from typing import Optional


def plot_conversion(
    df: pd.DataFrame,
    col: str,
    target_col: str        = "y",
    target_pos             = "yes",
    order: Optional[list]  = None,
    top_n: Optional[int]   = None,
    sort_by: str           = "conversion",   # "conversion" | "frequency" | "name" | None
    show_counts: bool      = True,
    show_avg_line: bool    = True,
    show_ci: bool          = True,
    figsize: tuple         = (10, 5),
    title: Optional[str]   = None,
    save_path: Optional[str] = None,
    ax: Optional[plt.Axes]   = None,
) -> plt.Axes:
    """
    Малює bar-chart конверсії (% позитивного класу) для кожної категорії в колонці.

    Параметри
    ----------
    df          : датафрейм
    col         : назва категоріальної колонки
    target_col  : назва цільової колонки (default: "y")
    target_pos  : значення позитивного класу (default: "yes")
    order       : фіксований порядок категорій (наприклад, місяці)
    top_n       : показати лише топ-N категорій за конверсією
    sort_by     : сортування — "conversion", "frequency", "name", або None
    show_counts : показувати кількість записів над стовпцями
    show_avg_line: показувати горизонтальну лінію середньої конверсії
    show_ci     : показувати 95% довірчий інтервал (Wilson interval)
    figsize     : розмір фігури
    title       : заголовок (якщо None — генерується автоматично)
    save_path   : шлях для збереження (якщо None — лише plt.show())
    ax          : існуючий Axes для вбудовування у субплот

    Повертає
    --------
    matplotlib Axes
    """

    # ── Підготовка даних ─────────────────────────────────────────────────────
    data = df[[col, target_col]].copy()
    data["_hit"] = (data[target_col] == target_pos).astype(int)

    grouped = (
        data.groupby(col)["_hit"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "n_pos", "count": "n_total"})
    )
    grouped["conv_pct"] = grouped["n_pos"] / grouped["n_total"] * 100

    # Wilson 95% довірчий інтервал
    def wilson_ci(n_pos, n_total, z=1.96):
        if n_total == 0:
            return 0.0, 0.0
        p    = n_pos / n_total
        denom = 1 + z**2 / n_total
        centre = (p + z**2 / (2 * n_total)) / denom
        half   = z * np.sqrt(p*(1-p)/n_total + z**2/(4*n_total**2)) / denom
        return max(0, (centre - half) * 100), min(100, (centre + half) * 100)

    grouped["ci_lo"], grouped["ci_hi"] = zip(*[
        wilson_ci(row.n_pos, row.n_total)
        for _, row in grouped.iterrows()
    ])
    grouped["ci_err_lo"] = grouped["conv_pct"] - grouped["ci_lo"]
    grouped["ci_err_hi"] = grouped["ci_hi"]    - grouped["conv_pct"]

    # ── Порядок категорій ─────────────────────────────────────────────────────
    if order is not None:
        grouped = grouped.reindex([x for x in order if x in grouped.index])
    elif sort_by == "conversion":
        grouped = grouped.sort_values("conv_pct", ascending=False)
    elif sort_by == "frequency":
        grouped = grouped.sort_values("n_total", ascending=False)
    elif sort_by == "name":
        grouped = grouped.sort_index()

    if top_n is not None:
        grouped = grouped.head(top_n)

    avg_conv = data["_hit"].mean() * 100
    n_cats   = len(grouped)

    # ── Кольори ───────────────────────────────────────────────────────────────
    colors = []
    for v in grouped["conv_pct"]:
        if v >= avg_conv * 1.5:
            colors.append("#27ae60")   # значно вище середнього
        elif v >= avg_conv * 0.85:
            colors.append("#4a90d9")   # близько до середнього
        else:
            colors.append("#e74c3c")   # нижче середнього

    # ── Малювання ─────────────────────────────────────────────────────────────
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_cats)

    yerr = np.array([
        grouped["ci_err_lo"].values,
        grouped["ci_err_hi"].values,
    ]) if show_ci else None

    bars = ax.bar(
        x, grouped["conv_pct"].values,
        color=colors, edgecolor="white", linewidth=0.8,
        yerr=yerr,
        error_kw=dict(ecolor="#555", capsize=4, lw=1.2, alpha=0.7) if show_ci else {},
        zorder=3,
    )

    # Середня лінія
    if show_avg_line:
        ax.axhline(avg_conv, color="#333", lw=1.3, ls="--", alpha=0.6, zorder=2,
                   label=f"Середня: {avg_conv:.1f}%")
        ax.legend(fontsize=9, loc="upper right")

    # Анотації над барами
    for bar, (_, row) in zip(bars, grouped.iterrows()):
        conv_txt = f"{row['conv_pct']:.1f}%"
        y_pos    = bar.get_height() + (row["ci_err_hi"] if show_ci else 0) + 0.5
        ax.annotate(
            conv_txt,
            (bar.get_x() + bar.get_width() / 2, y_pos),
            ha="center", va="bottom", fontsize=9, fontweight="500",
            color="#333",
        )
        if show_counts:
            ax.annotate(
                f"n={row['n_total']:,}",
                (bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (row["ci_err_hi"] if show_ci else 0) + 2.8),
                ha="center", va="bottom", fontsize=7.5, color="#777",
            )

    # Осі і підписи
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=35 if n_cats > 5 else 0,
                       ha="right" if n_cats > 5 else "center", fontsize=10)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))
    ax.set_ylabel("% підписок (yes)", fontsize=11)
    ax.set_xlabel(col, fontsize=11)
    ax.set_ylim(0, grouped["conv_pct"].max() * 1.35 + 3)
    ax.grid(axis="y", alpha=0.4, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Легенда кольорів
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor="#27ae60", label=f"≥ {avg_conv*1.5:.1f}% (значно вище)"),
        Patch(facecolor="#4a90d9", label=f"{avg_conv*0.85:.1f}–{avg_conv*1.5:.1f}% (норма)"),
        Patch(facecolor="#e74c3c", label=f"< {avg_conv*0.85:.1f}% (нижче середнього)"),
    ]
    ax.legend(handles=legend_els, fontsize=8, loc="upper right",
              framealpha=0.85, edgecolor="#ccc")

    auto_title = title or f"Конверсія по «{col}»  |  {target_pos} = позитивний клас"
    ax.set_title(auto_title, fontsize=12, fontweight="bold", pad=10)

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=130)
            print(f"Збережено: {save_path}")
        else:
            plt.show()

    return ax


# ── Демонстрація ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    df = pd.read_csv("/mnt/user-data/uploads/bank-additional-full.csv", sep=";")

    MONTH_ORDER = ["jan","feb","mar","apr","may","jun",
                   "jul","aug","sep","oct","nov","dec"]

    # Приклад 1 — job, сортування за конверсією (дефолт)
    plot_conversion(df, col="job",
                    save_path="/mnt/user-data/outputs/demo_job.png")

    # Приклад 2 — month, фіксований порядок місяців
    plot_conversion(df, col="month", order=MONTH_ORDER,
                    save_path="/mnt/user-data/outputs/demo_month.png")

    # Приклад 3 — poutcome, великий шрифт, без CI
    plot_conversion(df, col="poutcome", show_ci=False, figsize=(7, 5),
                    save_path="/mnt/user-data/outputs/demo_poutcome.png")

    # Приклад 4 — всі колонки на одній фігурі (subplot)
    cat_cols = ["job","marital","education","contact","poutcome"]
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(26, 6))
    fig.suptitle("Конверсія по категоріальних ознаках", fontsize=14, fontweight="bold")
    for ax, col in zip(axes, cat_cols):
        plot_conversion(df, col=col, ax=ax, show_counts=False,
                        title=col, figsize=(6, 5))
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/demo_all_subplots.png",
                bbox_inches="tight", dpi=130)

    print("Усі демо збережено.")
