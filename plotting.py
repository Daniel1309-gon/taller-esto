"""
plotting.py
===========
Utilidades para generar y guardar las gráficas PNG y los resúmenes CSV de los
experimentos de simulación definidos en main.py.

Funciones exportadas
--------------------
slugify(text)
    Convierte un título legible en un nombre de archivo portable (sin acentos
    ni caracteres especiales).

save_line_plot(...)
    Gráfica de líneas simple (sin barras de error).  No se usa directamente en
    main.py, pero está disponible para exploración rápida de resultados crudos.

summarize_results(df, group_cols)
    Agrega media, desviación estándar y conteo de P(True) y Time(s) agrupando
    por las columnas indicadas.  Devuelve un DataFrame plano listo para CSV.

save_summary_csv(summary, filename)
    Serializa el resumen estadístico en OUTPUT_DIR/<filename>.

save_errorbar_plot(...)
    Gráfica de media ± desviación estándar (errorbar).  Es la visualización
    principal para comparar algoritmos entre corridas independientes.

plot_experiment_results(scenario_name, df)
    Orquesta resumen + dos gráficas para los Escenarios 1, 2 y 3.

plot_burn_in_results(scenario_name, df)
    Orquesta resumen + dos gráficas para el Escenario 4 (efecto del burn-in).

plot_rare_evidence_results(scenario_name, df)
    Orquesta resumen + dos gráficas para el Escenario 5 (evidencia extrema).

Todos los archivos generados se depositan en el directorio OUTPUT_DIR ("outputs/"),
que se crea automáticamente si no existe.
"""

import re
import unicodedata
from pathlib import Path

import matplotlib

# "Agg" es el backend de renderizado no interactivo de Matplotlib.
# No requiere pantalla ni entorno gráfico (X11, Wayland, etc.), por lo que
# funciona en servidores y scripts desatendidos.  Debe llamarse ANTES del primer
# import de matplotlib.pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Directorio donde se depositan todas las gráficas PNG y los CSVs de resumen.
# Se crea automáticamente la primera vez que se llama a mkdir(exist_ok=True).
OUTPUT_DIR = Path("outputs")


def slugify(text):
    """
    Convierte un título de escenario en un nombre de archivo estable y portable.

    Pipeline de transformación
    --------------------------
    1. Normalización NFKD: descompone caracteres Unicode en su forma canónica
       (p.ej. "é" → "e" + combinador de acento) para poder separar la letra
       base de su diacrítico.
    2. Codificación ASCII + ignore: descarta los diacríticos y cualquier
       carácter no representable en ASCII (p.ej. 'ñ', 'ü').
    3. Minúsculas: hace el nombre de archivo insensible a capitalización.
    4. Sustitución: reemplaza secuencias de caracteres que no son letras ni
       dígitos (espacios, paréntesis, guiones, etc.) con un guion bajo.
    5. Strip: elimina guiones bajos al inicio y al final del resultado.

    Ejemplo
    -------
    "Escenario 1 (Evidencia Única)" → "escenario_1_evidencia_unica"

    Parámetros
    ----------
    text : str
        Título legible del escenario (puede contener acentos y símbolos).

    Devuelve
    --------
    str
        Nombre de archivo válido en Windows, macOS y Linux.
    """
    # Paso 1: descomposición Unicode → separar letra base de diacrítico
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    # Paso 2-3: minúsculas
    normalized = normalized.lower()
    # Paso 4: cualquier secuencia de caracteres "no-slug" se convierte en "_"
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    # Paso 5: limpiar guiones bajos en los bordes
    return normalized.strip("_")


def save_line_plot(df, x_col, y_col, title, x_label, y_label, filename, group_col=None, log_x=False):
    """
    Renderiza y guarda una gráfica de líneas en OUTPUT_DIR/<filename>.

    Cuando se especifica group_col, dibuja una línea por cada valor distinto de
    esa columna (p.ej. una línea por algoritmo).  Si no se especifica, dibuja
    una única línea con todos los datos.

    Nota: esta función grafica los datos crudos (sin promediar); para comparar
    corridas múltiples con barras de error usa save_errorbar_plot en su lugar.

    Parámetros
    ----------
    df : pd.DataFrame
        Datos a graficar (filas = observaciones individuales).
    x_col : str
        Columna que define el eje X (p.ej. "N" o "Burn-in").
    y_col : str
        Columna que define el eje Y (p.ej. "P(True)" o "Time(s)").
    title : str
        Título que aparece en la parte superior de la figura.
    x_label : str
        Etiqueta del eje X.
    y_label : str
        Etiqueta del eje Y.
    filename : str
        Nombre del archivo PNG de salida (sin ruta; se guarda en OUTPUT_DIR).
    group_col : str, opcional
        Columna usada para separar en múltiples líneas (p.ej. "Algorithm").
    log_x : bool
        Si True, el eje X se muestra en escala logarítmica.  Recomendado cuando
        los valores de N varían en órdenes de magnitud (100, 1000, 10000, ...).
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    # figsize en pulgadas: 8×5 es una proporción cómoda para informes académicos
    fig, ax = plt.subplots(figsize=(8, 5))

    if group_col:
        for group_name, group_df in df.groupby(group_col):
            # Ordenar por X para que la línea sea continua y no dibuje "zigzag"
            group_df = group_df.sort_values(x_col)
            ax.plot(group_df[x_col], group_df[y_col], marker="o", linewidth=2, label=group_name)
        ax.legend(title=group_col)
    else:
        ordered_df = df.sort_values(x_col)
        ax.plot(ordered_df[x_col], ordered_df[y_col], marker="o", linewidth=2)

    if log_x:
        # Escala log en X resalta diferencias entre N pequeños sin comprimir los grandes
        ax.set_xscale("log")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # Cuadrícula semitransparente para facilitar la lectura sin distraer
    ax.grid(True, alpha=0.3)
    # Ajusta márgenes automáticamente para que etiquetas y leyenda no se recorten
    fig.tight_layout()

    output_path = OUTPUT_DIR / filename
    # dpi=150: resolución suficiente para informes PDF e impresión A4 sin pixelado
    fig.savefig(output_path, dpi=150)
    # Cerrar la figura libera la memoria del backend Agg (importante en bucles)
    plt.close(fig)
    print(f"Gráfica guardada: {output_path}")


def summarize_results(df, group_cols):
    """
    Agrega los resultados crudos calculando media, desviación estándar y conteo
    de P(True) —y opcionalmente de Time(s)— agrupando por las columnas indicadas.

    El DataFrame devuelto tiene columnas "planas" (sin MultiIndex) listas para
    imprimirse en consola o guardarse como CSV.

    Parámetros
    ----------
    df : pd.DataFrame
        Resultados crudos del experimento.  Debe tener al menos las columnas
        presentes en group_cols y la columna "P(True)".
    group_cols : list[str]
        Columnas por las que se agrupa antes de calcular estadísticos.
        Ejemplos: ["N", "Algorithm"] para escenarios estándar,
                  ["Burn-in"] para el experimento de burn-in.

    Devuelve
    --------
    pd.DataFrame
        Tabla con una fila por combinación única de group_cols y las columnas:
        Runs, P(True) mean, P(True) std, [Time(s) mean, Time(s) std].
    """
    aggregations = {
        # mean y std para comparar precisión; count para verificar cuántas corridas
        # entraron en cada celda (útil como auditoría de que num_runs es correcto).
        "P(True)": ["mean", "std", "count"],
    }
    if "Time(s)" in df.columns:
        # El tiempo de ejecución es opcional: solo se agrega si la columna existe
        aggregations["Time(s)"] = ["mean", "std"]

    summary = df.groupby(group_cols, as_index=False).agg(aggregations)

    # pandas genera columnas MultiIndex del tipo ("P(True)", "mean") al usar agg()
    # con múltiples funciones.  Las aplanamos a strings "P(True) mean" para que
    # el DataFrame sea más fácil de leer e indexar.
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
    # Cuando solo hay 1 corrida, la std es NaN (no definida); se sustituye por 0
    # para que las gráficas de errorbar no fallen al intentar trazar NaN.
    return summary.fillna(0.0)


def save_summary_csv(summary, filename):
    """
    Serializa el DataFrame de resumen estadístico en OUTPUT_DIR/<filename>.

    Los archivos CSV permiten importar los resultados en hojas de cálculo,
    LaTeX (via pandas.read_csv + to_latex) o herramientas de análisis externas
    sin necesidad de re-ejecutar los experimentos.

    Parámetros
    ----------
    summary : pd.DataFrame
        Tabla de estadísticos (salida de summarize_results).
    filename : str
        Nombre del archivo CSV de salida (sin ruta; se guarda en OUTPUT_DIR).
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / filename
    # index=False: omite la columna de índice de pandas en el CSV para mayor limpieza
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
    y_limits=None,
):
    """
    Renderiza y guarda una gráfica de media ± desviación estándar (errorbar).

    Las barras de error representan ±1σ calculado sobre las corridas independientes
    de cada configuración (N, Algorithm).  Permiten visualizar de un vistazo si un
    algoritmo converge de forma estable o presenta alta variabilidad entre corridas:
      - Barras cortas → estimador estable y reproducible.
      - Barras largas → alta varianza, posiblemente N insuficiente o mala mezcla
        de la cadena de Markov (en Gibbs Sampling).

    Parámetros
    ----------
    summary : pd.DataFrame
        DataFrame de resumen (salida de summarize_results).
    x_col : str
        Columna del eje X (p.ej. "N" o "Burn-in").
    mean_col : str
        Columna con la media a graficar (p.ej. "P(True) mean").
    std_col : str
        Columna con la desviación estándar para las barras de error
        (p.ej. "P(True) std").
    title : str
        Título de la figura.
    x_label : str
        Etiqueta del eje X.
    y_label : str
        Etiqueta del eje Y.
    filename : str
        Nombre del archivo PNG de salida (sin ruta; se guarda en OUTPUT_DIR).
    group_col : str, opcional
        Columna para separar en múltiples líneas (p.ej. "Algorithm").  Si no
        se especifica se dibuja una única línea.
    log_x : bool
        Si True, eje X en escala logarítmica.
    y_limits : tuple (float, float), opcional
        Límites (min, max) del eje Y.  Fijar (0, 1) para P(True) facilita
        comparar escenarios en la misma escala visual.
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
                capsize=4,       # anchura (en puntos) de las tapas horizontales de las barras
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
    if y_limits:
        # Desempaquetar la tupla (min, max) directamente en set_ylim
        ax.set_ylim(*y_limits)

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
    Genera y guarda las gráficas y el resumen CSV para un escenario estándar
    (Escenarios 1, 2 y 3 de main.py: evidencia única, múltiple y causa rara).

    Archivos generados en OUTPUT_DIR
    --------------------------------
    - <slug>_resumen.csv        : media y std de P(True) y Time(s) por (N, Algorithm).
    - <slug>_probabilidad.png  : P(True) media ± σ vs N, una línea por algoritmo.
    - <slug>_tiempo.png        : Tiempo de ejecución medio ± σ vs N, una línea por
                                  algoritmo.  Permite comparar el costo computacional.

    Ambas gráficas usan escala logarítmica en X para que los saltos de N sean
    equidistantes visualmente (100 → 1000 → 10000 → ...).

    Parámetros
    ----------
    scenario_name : str
        Nombre del escenario; se convierte en slug para los nombres de archivo.
    df : pd.DataFrame
        Resultados crudos con columnas: Run, N, Algorithm, P(True), Time(s).

    Devuelve
    --------
    pd.DataFrame
        Tabla de resumen estadístico (la misma que se guarda en CSV).
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
        y_limits=(0, 1),
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
    Genera y guarda las gráficas y el resumen CSV para el experimento de burn-in
    (Escenario 4 de main.py).

    A diferencia de plot_experiment_results, el eje X es "Burn-in" (no "N") y no
    se agrupa por algoritmo porque este escenario solo evalúa Gibbs Sampling con N
    de conteo fijo y distintos períodos de calentamiento.

    El objetivo es observar a partir de qué valor de burn-in la estimación de
    P(True) se estabiliza: con burn_in = 0 el transiente inicial sesga el resultado;
    al aumentar burn_in la curva converge a un valor estable.

    Archivos generados en OUTPUT_DIR
    --------------------------------
    - <slug>_resumen.csv        : media y std de P(True) y Time(s) por Burn-in.
    - <slug>_probabilidad.png  : P(True) media ± σ vs Burn-in.
    - <slug>_tiempo.png        : Tiempo de ejecución medio ± σ vs Burn-in.

    Parámetros
    ----------
    scenario_name : str
        Nombre del escenario.
    df : pd.DataFrame
        Resultados con columnas: Run, Burn-in, N conteo, P(True), Time(s).

    Devuelve
    --------
    pd.DataFrame
        Tabla de resumen estadístico.
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
        y_limits=(0, 1),
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
    Genera y guarda las gráficas y el resumen CSV para el escenario de evidencia
    extrema (Escenario 5 de main.py).

    Idéntico en estructura a plot_experiment_results, pero orientado a evidenciar
    el comportamiento de Rejection Sampling ante evidencia combinada de probabilidad
    prior muy baja:
      - Para N pequeños, Rejection puede retornar P(True) = 0.5 (fallback por falta
        de muestras aceptadas), visible como una línea plana en la gráfica.
      - Para N grandes, los tres algoritmos convergen al mismo valor verdadero.

    El límite del eje Y fijado en (0, 1) facilita comparar la amplitud de las
    barras de error entre Rejection, Likelihood Weighting y Gibbs.

    Archivos generados en OUTPUT_DIR
    --------------------------------
    - <slug>_resumen.csv        : media y std de P(True) y Time(s) por (N, Algorithm).
    - <slug>_probabilidad.png  : P(True) media ± σ vs N, una línea por algoritmo.
    - <slug>_tiempo.png        : Tiempo medio ± σ vs N, una línea por algoritmo.

    Parámetros
    ----------
    scenario_name : str
        Nombre del escenario.
    df : pd.DataFrame
        Resultados crudos con columnas: Run, N, Algorithm, P(True), Time(s).

    Devuelve
    --------
    pd.DataFrame
        Tabla de resumen estadístico.
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
        y_limits=(0, 1),
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
