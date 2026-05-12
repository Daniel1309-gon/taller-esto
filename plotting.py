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


def plot_experiment_results(scenario_name, df):
    """
    Genera las gráficas de probabilidad y tiempo para un escenario estándar.
    """
    base_name = slugify(scenario_name)
    save_line_plot(
        df,
        x_col="N",
        y_col="P(True)",
        title=f"{scenario_name} - Probabilidad estimada",
        x_label="N (muestras)",
        y_label="P(True)",
        filename=f"{base_name}_probabilidad.png",
        group_col="Algorithm",
        log_x=True,
    )
    save_line_plot(
        df,
        x_col="N",
        y_col="Time(s)",
        title=f"{scenario_name} - Tiempo de ejecución",
        x_label="N (muestras)",
        y_label="Tiempo (s)",
        filename=f"{base_name}_tiempo.png",
        group_col="Algorithm",
        log_x=True,
    )


def plot_burn_in_results(scenario_name, df):
    """
    Genera las gráficas de probabilidad y tiempo para el experimento de burn-in.
    """
    base_name = slugify(scenario_name)
    save_line_plot(
        df,
        x_col="Burn-in",
        y_col="P(True)",
        title=f"{scenario_name} - Probabilidad estimada",
        x_label="Burn-in",
        y_label="P(True)",
        filename=f"{base_name}_probabilidad.png",
    )
    save_line_plot(
        df,
        x_col="Burn-in",
        y_col="Time(s)",
        title=f"{scenario_name} - Tiempo de ejecución",
        x_label="Burn-in",
        y_label="Tiempo (s)",
        filename=f"{base_name}_tiempo.png",
    )


def plot_rare_evidence_results(scenario_name, df):
    """
    Genera la gráfica de probabilidad para el escenario de evidencia extrema.
    """
    base_name = slugify(scenario_name)
    save_line_plot(
        df,
        x_col="N",
        y_col="P(True)",
        title=f"{scenario_name} - Probabilidad estimada",
        x_label="N (muestras)",
        y_label="P(True)",
        filename=f"{base_name}_probabilidad.png",
        group_col="Algorithm",
        log_x=True,
    )
