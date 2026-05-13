"""
plotting.py
===========
Utilidades para generar las gráficas PNG de los experimentos de simulación.
"""

import re
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("outputs")


def slugify(text):
    """
    Convierte un título de escenario en un nombre de archivo estable.
    """
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def save_line_plot(df, x_col, y_col, title, x_label, y_label, filename, group_col=None, log_x=False):
    """
    Guarda una gráfica de líneas a partir de un DataFrame de resultados.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    if group_col:
        for group_name, group_df in df.groupby(group_col):
            group_df = group_df.sort_values(x_col)
            ax.plot(group_df[x_col], group_df[y_col], marker="o", linewidth=2, label=group_name)
        ax.legend(title=group_col)
    else:
        ordered_df = df.sort_values(x_col)
        ax.plot(ordered_df[x_col], ordered_df[y_col], marker="o", linewidth=2)

    if log_x:
        ax.set_xscale("log")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Gráfica guardada: {output_path}")


def summarize_results(df, group_cols):
    """
    Calcula media y desviación estándar de probabilidad y tiempo por configuración.
    """
    aggregations = {
        "P(True)": ["mean", "std", "count"],
    }
    if "Time(s)" in df.columns:
        aggregations["Time(s)"] = ["mean", "std"]

    summary = df.groupby(group_cols, as_index=False).agg(aggregations)
    summary.columns = [
        " ".join(col).strip() if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary = summary.rename(columns={
        "P(True) count": "Runs",
        "P(True) mean": "P(True) mean",
        "P(True) std": "P(True) std",
        "Time(s) mean": "Time(s) mean",
        "Time(s) std": "Time(s) std",
    })
    return summary.fillna(0.0)


def save_summary_csv(summary, filename):
    """
    Guarda una tabla resumen con media y desviación estándar.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / filename
    summary.to_csv(output_path, index=False)
    print(f"Resumen guardado: {output_path}")


def save_errorbar_plot(
    summary,
    x_col,
    mean_col,
    std_col,
    title,
    x_label,
    y_label,
    filename,
    group_col=None,
    log_x=False,
):
    """
    Guarda una gráfica de media con barras de desviación estándar.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    if group_col:
        for group_name, group_df in summary.groupby(group_col):
            group_df = group_df.sort_values(x_col)
            ax.errorbar(
                group_df[x_col],
                group_df[mean_col],
                yerr=group_df[std_col],
                marker="o",
                linewidth=2,
                capsize=4,
                label=group_name,
            )
        ax.legend(title=group_col)
    else:
        ordered_df = summary.sort_values(x_col)
        ax.errorbar(
            ordered_df[x_col],
            ordered_df[mean_col],
            yerr=ordered_df[std_col],
            marker="o",
            linewidth=2,
            capsize=4,
        )

    if log_x:
        ax.set_xscale("log")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Gráfica guardada: {output_path}")


def plot_experiment_results(scenario_name, df):
    """
    Genera gráficas y resumen estadístico para un escenario estándar.
    """
    base_name = slugify(scenario_name)
    summary = summarize_results(df, ["N", "Algorithm"])
    save_summary_csv(summary, f"{base_name}_resumen.csv")
    save_errorbar_plot(
        summary,
        x_col="N",
        mean_col="P(True) mean",
        std_col="P(True) std",
        title=f"{scenario_name} - P(True) media +/- desviación estándar",
        x_label="N (muestras)",
        y_label="P(True)",
        filename=f"{base_name}_probabilidad.png",
        group_col="Algorithm",
        log_x=True,
    )
    save_errorbar_plot(
        summary,
        x_col="N",
        mean_col="Time(s) mean",
        std_col="Time(s) std",
        title=f"{scenario_name} - Tiempo medio +/- desviación estándar",
        x_label="N (muestras)",
        y_label="Tiempo (s)",
        filename=f"{base_name}_tiempo.png",
        group_col="Algorithm",
        log_x=True,
    )
    return summary


def plot_burn_in_results(scenario_name, df):
    """
    Genera gráficas y resumen estadístico para el experimento de burn-in.
    """
    base_name = slugify(scenario_name)
    summary = summarize_results(df, ["Burn-in"])
    save_summary_csv(summary, f"{base_name}_resumen.csv")
    save_errorbar_plot(
        summary,
        x_col="Burn-in",
        mean_col="P(True) mean",
        std_col="P(True) std",
        title=f"{scenario_name} - P(True) media +/- desviación estándar",
        x_label="Burn-in",
        y_label="P(True)",
        filename=f"{base_name}_probabilidad.png",
    )
    save_errorbar_plot(
        summary,
        x_col="Burn-in",
        mean_col="Time(s) mean",
        std_col="Time(s) std",
        title=f"{scenario_name} - Tiempo medio +/- desviación estándar",
        x_label="Burn-in",
        y_label="Tiempo (s)",
        filename=f"{base_name}_tiempo.png",
    )
    return summary


def plot_rare_evidence_results(scenario_name, df):
    """
    Genera gráficas y resumen estadístico para el escenario de evidencia extrema.
    """
    base_name = slugify(scenario_name)
    summary = summarize_results(df, ["N", "Algorithm"])
    save_summary_csv(summary, f"{base_name}_resumen.csv")
    save_errorbar_plot(
        summary,
        x_col="N",
        mean_col="P(True) mean",
        std_col="P(True) std",
        title=f"{scenario_name} - P(True) media +/- desviación estándar",
        x_label="N (muestras)",
        y_label="P(True)",
        filename=f"{base_name}_probabilidad.png",
        group_col="Algorithm",
        log_x=True,
    )
    save_errorbar_plot(
        summary,
        x_col="N",
        mean_col="Time(s) mean",
        std_col="Time(s) std",
        title=f"{scenario_name} - Tiempo medio +/- desviación estándar",
        x_label="N (muestras)",
        y_label="Tiempo (s)",
        filename=f"{base_name}_tiempo.png",
        group_col="Algorithm",
        log_x=True,
    )
    return summary
