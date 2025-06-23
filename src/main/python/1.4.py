import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm
import glob
from collections import defaultdict
import argparse
import os

# Simulation constants
L = 0.1  # Container diameter (m)
R = 0.005  # Obstacle radius (m)
CONTAINER_RADIUS = L / 2  # 0.05 m
OBSTACLE_RADIUS = R  # 0.005 m


def find_obstacle_wall_time(files, particle_id=0):
    """
    Find the time when the obstacle reaches the wall by analyzing simulation data.
    Returns the minimum time across all files where the obstacle reaches the wall boundary.
    """
    wall_times = []

    for file in files:
        times, positions = load_positions(file, particle_id)

        if len(times) == 0 or len(positions) == 0:
            continue

        # Calculate radial distance from center for each position
        radial_distances = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)

        # Find when obstacle reaches wall boundary (container_radius - obstacle_radius)
        wall_boundary = CONTAINER_RADIUS - OBSTACLE_RADIUS  # 0.045 m

        # Find first time when obstacle reaches or exceeds wall boundary
        wall_reached = radial_distances >= wall_boundary

        if np.any(wall_reached):
            # Find the first occurrence
            first_wall_time_idx = np.where(wall_reached)[0][0]
            wall_time = times[first_wall_time_idx]
            wall_times.append(wall_time)
            print(f"  {file}: Obstacle reaches wall at t = {wall_time:.4f} s")
        else:
            print(
                f"  {file}: Obstacle never reaches wall (max distance: {np.max(radial_distances):.4f} m)"
            )

    if not wall_times:
        print(
            "Warning: Obstacle never reaches wall in any file. Using full simulation time."
        )
        return None

    # Return the minimum time across all files
    min_wall_time = min(wall_times)
    print(f"Earliest wall collision time across all files: {min_wall_time:.4f} s")
    return min_wall_time


def analyze_simulation_timing(files, particle_id=0):
    """
    Analyze the time intervals between simulation outputs to justify dt_bin choice.
    """
    print("\n=== ANÁLISIS DE INTERVALOS TEMPORALES ===")
    all_intervals = []

    for file in files:
        times, _ = load_positions(file, particle_id)
        if len(times) > 1:
            intervals = np.diff(times)  # Time between consecutive saves
            all_intervals.extend(intervals)

    if not all_intervals:
        print("No hay datos suficientes para analizar intervalos")
        return

    intervals = np.array(all_intervals)

    print(f"Estadísticas de intervalos entre eventos de simulación:")
    print(f"  Número total de intervalos: {len(intervals)}")
    print(f"  Media: {np.mean(intervals):.6f} s")
    print(f"  Mediana: {np.median(intervals):.6f} s")
    print(f"  Mínimo: {np.min(intervals):.6f} s")
    print(f"  Máximo: {np.max(intervals):.6f} s")
    print(f"  Desviación estándar: {np.std(intervals):.6f} s")
    print(f"  Percentil 25: {np.percentile(intervals, 25):.6f} s")
    print(f"  Percentil 75: {np.percentile(intervals, 75):.6f} s")
    print(f"  Percentil 95: {np.percentile(intervals, 95):.6f} s")

    return intervals


def calculate_dcm_with_dt(files, particle_id, dt_bin, t_max):
    """
    Calculate DCM for a specific dt_bin value and return statistics.
    """
    dcms_by_time = defaultdict(list)

    for file in files:
        times, positions = load_positions(file, particle_id)

        if len(times) == 0 or len(positions) == 0:
            continue

        # Apply time limits
        mask = (times >= 0) & (times <= t_max)
        times = times[mask]
        positions = positions[mask]

        if len(times) == 0:
            continue

        # Calculate DCM from initial position
        initial_pos = positions[0]
        dcm = np.sum((positions - initial_pos) ** 2, axis=1)

        # Group by time bins
        binned = group_by_time_bin(times, dcm, dt_bin=dt_bin)
        for t_bin, z2 in binned.items():
            dcms_by_time[t_bin].extend(z2 if isinstance(z2, list) else [z2])

    # Sort and compute stats
    sorted_times = sorted(dcms_by_time.keys())
    mean_dcm = [np.mean(dcms_by_time[t]) for t in sorted_times]
    std_dcm = [np.std(dcms_by_time[t]) for t in sorted_times]

    return np.array(sorted_times), np.array(mean_dcm), np.array(std_dcm)


def dt_sensitivity_analysis(files, particle_id, t_max):
    """
    Perform sensitivity analysis for different dt_bin values.
    """
    print("\n=== ANÁLISIS DE SENSIBILIDAD DE dt_bin ===")

    # Test different dt_bin values
    dt_values = np.logspace(-5, -2, 20)  # From 1e-5 to 1e-2
    dt_stds = []
    dt_results = []

    for dt_bin in dt_values:
        try:
            times, mean_dcm, std_dcm = calculate_dcm_with_dt(
                files, particle_id, dt_bin, t_max
            )

            if len(times) > 10:  # Need sufficient data points
                # Calculate overall standard deviation of DCM values
                overall_std = np.std(mean_dcm)
                dt_stds.append(overall_std)
                dt_results.append((dt_bin, overall_std, len(times)))
                print(
                    f"dt_bin = {dt_bin:.6f} s: std = {overall_std:.6e} m², puntos = {len(times)}"
                )
            else:
                dt_stds.append(np.inf)
                dt_results.append((dt_bin, np.inf, len(times)))

        except Exception as e:
            print(f"Error con dt_bin = {dt_bin:.6f} s: {e}")
            dt_stds.append(np.inf)
            dt_results.append((dt_bin, np.inf, 0))

    dt_stds = np.array(dt_stds)

    # Find optimal dt_bin
    valid_mask = np.isfinite(dt_stds)
    if np.any(valid_mask):
        min_idx = np.argmin(dt_stds[valid_mask])
        optimal_dt = dt_values[valid_mask][min_idx]
        optimal_std = dt_stds[valid_mask][min_idx]

        print(f"\ndt_bin óptimo encontrado: {optimal_dt:.6f} s")
        print(f"Desviación estándar mínima: {optimal_std:.6e} m²")
    else:
        optimal_dt = 0.001  # Fallback
        optimal_std = np.inf
        print("\nNo se pudo encontrar dt_bin óptimo, usando valor por defecto")

    # Plot std vs dt_bin
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(format_func))
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))
    valid_dt = dt_values[valid_mask]
    valid_std = dt_stds[valid_mask]

    plt.loglog(valid_dt, valid_std, "o-", color="blue", linewidth=2, markersize=6)

    # Format optimal_dt using the same format_func
    optimal_dt_formatted = format_func(optimal_dt, None)

    plt.axvline(
        x=optimal_dt,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"dt óptimo = {optimal_dt_formatted} s",
    )
    plt.xlabel("dt [s]", fontsize=12)
    plt.ylabel("Desviación estándar DCM [m²]", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    sensitivity_plot_filename = "analysis/dt_sensitivity_analysis.png"
    plt.savefig(sensitivity_plot_filename, dpi=300, bbox_inches="tight")
    print(f"Gráfico de sensibilidad guardado en: {sensitivity_plot_filename}")

    return optimal_dt, dt_results


def load_positions(filename, particle_id=0):
    """Parse simulation output file and extract particle positions over time."""
    with open(filename, "r") as f:
        content = f.read()

    blocks = content.split("---")
    times, positions = [], []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        time = float(lines[0])
        times.append(time)

        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            idx, x, y, vx, vy = map(float, parts)
            if int(idx) == particle_id:  # Track specified particle
                positions.append((x, y))
                break

    return np.array(times), np.array(positions)


def group_by_time_bin(times, values, dt_bin=0.001):
    binned = defaultdict(list)
    for t, v in zip(times, values):
        t_bin = round(t / dt_bin) * dt_bin
        binned[t_bin].append(v)
    return binned


def format_func(value, _):
    if value == 0 or np.isclose(value, 0):
        return "0"
    try:
        exp = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10**exp)
        if exp == 0:
            return f"{mantissa:.1f}"
        else:
            return rf"${mantissa:.1f} \times 10^{{{exp}}}$"
    except:
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Análisis de DCM para partícula grande (M=3kg) - Event-Driven MD"
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="Tiempo máximo para el análisis (s) - si no se especifica, se calcula automáticamente",
    )
    parser.add_argument(
        "--dt-bin",
        type=float,
        default=0.001,
        help="Intervalo de tiempo para agrupación (s) - debe ser consistente con el cálculo DCM",
    )
    parser.add_argument(
        "--particle-id",
        type=int,
        default=0,
        help="ID de la partícula grande a analizar (default: 0)",
    )
    parser.add_argument(
        "--optimize-dt",
        action="store_true",
        help="Realizar análisis de sensibilidad para encontrar dt_bin óptimo (más lento)",
    )
    args = parser.parse_args()

    # Main processing
    dt_bin = args.dt_bin
    all_files = sorted(glob.glob("output/*"))

    dcms_by_time = defaultdict(list)
    max_times = []

    print(f"Analizando partícula ID: {args.particle_id} (M=3kg)")
    print(f"dt_bin para agrupación: {dt_bin:.6f} s")
    print(f"Archivos encontrados: {len(all_files)}")

    # Analyze simulation timing to justify dt_bin choice
    intervals = analyze_simulation_timing(all_files, args.particle_id)

    # Calculate maximum time automatically if not provided
    if args.tmax is None:
        print("Calculando tiempo máximo automáticamente...")
        t_max = find_obstacle_wall_time(all_files, args.particle_id)
    else:
        t_max = args.tmax
        print(f"Usando tiempo máximo especificado: {t_max:.4f} s")

    # First pass: compute minimum common time
    max_times = []
    for file in all_files:
        times, _ = load_positions(file, args.particle_id)
        if len(times) > 0:
            max_times.append(times[-1])

    if not max_times:
        print("Error: No valid data found in any files")
        exit()

    t_min = max(max_times)

    # Use the automatically calculated time if --tmax was not provided
    if args.tmax is None:
        if t_max is not None:
            t_max = min(
                t_max, t_min
            )  # Use the smaller of calculated wall time or available data
        else:
            t_max = t_min  # If no wall collision found, use full available time
    else:
        t_max = args.tmax

    print(f"Tiempo de corte (mínimo entre todas las corridas): {t_min:.4f} s")
    if args.tmax is not None:
        print(f"Tiempo máximo limitado a: {t_max:.4f} s")
    else:
        print(f"Tiempo máximo calculado automáticamente: {t_max:.4f} s")

    # Conditionally perform dt sensitivity analysis
    if args.optimize_dt:
        optimal_dt, dt_analysis_results = dt_sensitivity_analysis(
            all_files, args.particle_id, t_max
        )
        print(
            f"\nUsando dt_bin óptimo: {optimal_dt:.6f} s (en lugar de {dt_bin:.6f} s)"
        )
        dt_bin = optimal_dt

        # Calculate DCM with optimal dt_bin
        sorted_times, mean_dcm, std_dcm = calculate_dcm_with_dt(
            all_files, args.particle_id, dt_bin, t_max
        )
    else:
        print(
            f"\nUsando dt_bin especificado: {dt_bin:.6f} s (para optimizar usar --optimize-dt)"
        )

        # Calculate DCM with specified dt_bin (original method)
        dcms_by_time = defaultdict(list)

        for file in all_files:
            times, positions = load_positions(file, args.particle_id)

            if len(times) == 0 or len(positions) == 0:
                print(f"Warning: No data found in {file}")
                continue

            # Apply time limits
            mask = (times >= 0) & (times <= t_max)
            times = times[mask]
            positions = positions[mask]

            if len(times) == 0:
                print(f"Warning: No data within time range for {file}")
                continue

            # Calculate DCM from initial position
            initial_pos = positions[0]
            dcm = np.sum((positions - initial_pos) ** 2, axis=1)

            # Group by time bins for ensemble averaging
            binned = group_by_time_bin(times, dcm, dt_bin=dt_bin)
            for t_bin, z2 in binned.items():
                dcms_by_time[t_bin].extend(z2 if isinstance(z2, list) else [z2])

        # Sort and compute stats
        sorted_times = sorted(dcms_by_time.keys())
        mean_dcm = [np.mean(dcms_by_time[t]) for t in sorted_times]
        std_dcm = [np.std(dcms_by_time[t]) for t in sorted_times]

        # Convert to numpy arrays
        sorted_times = np.array(sorted_times)
        mean_dcm = np.array(mean_dcm)
        std_dcm = np.array(std_dcm)

    print(f"Datos cargados y agrupados: {len(sorted_times)} elementos")
    print(f"Rango temporal: {sorted_times[0]:.4f} - {sorted_times[-1]:.4f} s")

    # Linear fit for diffusion coefficient (forced through origin)
    # DCM = 4*D*t, so we fit y = mx where m = 4*D
    slope = np.sum(sorted_times * mean_dcm) / np.sum(sorted_times**2)
    D = slope / 4  # D = slope/4 for 2D diffusion
    fit_line = slope * sorted_times  # y = mx (no intercept)

    # Calculate R² and residuals for fit quality
    residuals = np.array(mean_dcm) - np.array(fit_line)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((mean_dcm - np.mean(mean_dcm)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1
    residual_std = np.std(residuals)

    # Error analysis for E(D) plot
    optimal_slope = slope
    optimal_D = D

    # Create a range of D values to test around the optimal one
    d_range = np.linspace(optimal_D * 0.5, optimal_D * 1.5, 200)

    # Calculate the error E(D) for each D value
    # E(D) = sum_i [y_i - f(x_i, D)]^2, where f(x_i, D) = (4*D)*x_i (no intercept)
    fit_errors = []
    for d_val in d_range:
        current_slope = 4 * d_val
        predicted_dcm = current_slope * sorted_times  # No intercept
        error = np.sum((np.array(mean_dcm) - predicted_dcm) ** 2)
        fit_errors.append(error)

    print(f"Coeficiente de difusión: D = {D:.6e} m²/s")
    print(f"R² del ajuste: {r_squared:.6f}")

    # Set style for better looking plots
    plt.style.use("default")

    # --- PLOT 1: DCM vs Time ---
    fig_dcm, ax_dcm = plt.subplots(figsize=(12, 8))

    ax_dcm.fill_between(
        sorted_times,
        np.array(mean_dcm) - np.array(std_dcm),
        np.array(mean_dcm) + np.array(std_dcm),
        color="#2E86AB",
        alpha=0.2,
        label="±1σ",
    )

    ax_dcm.plot(
        sorted_times,
        mean_dcm,
        color="#2E86AB",
        linewidth=1.2,
        alpha=0.9,
        label="DCM (Ensemble)",
    )

    ax_dcm.plot(
        sorted_times,
        fit_line,
        color="#A23B72",
        linewidth=3,
        label=f"Ajuste Lineal (D={D:.2e} m²/s)",
    )

    title_text_dcm = f"Desplazamiento Cuadrático Medio - Partícula Grande (M=3kg)\n{len(all_files)} archivos, dt={dt_bin:.3f}s"
    if args.tmax is not None:
        title_text_dcm += f"\nRango: 0.0 - {t_max:.2f} s"
    else:
        title_text_dcm += f"\nRango: 0.0 - {t_max:.2f} s (calculado automáticamente)"

    # ax_dcm.set_title(title_text_dcm, fontsize=16, fontweight='bold', pad=20)
    ax_dcm.set_xlabel("Tiempo (s)", fontsize=14)
    ax_dcm.set_ylabel("DCM (m²)", fontsize=14)

    ax_dcm.legend(
        loc="upper left", fontsize=12, framealpha=0.9, fancybox=True, shadow=True
    )
    ax_dcm.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax_dcm.set_ylim(bottom=0)

    # stats_text_dcm = f"R² = {r_squared:.4f}\nN = {len(all_files)} archivos\nPuntos = {len(sorted_times)}\nD = {D:.2e} m²/s"
    # ax_dcm.text(0.02, 0.98, stats_text_dcm, transform=ax_dcm.transAxes,
    #        verticalalignment='top', fontsize=11,
    #        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax_dcm.tick_params(axis="both", which="major", labelsize=12)
    ax_dcm.yaxis.set_major_formatter(FuncFormatter(format_func))
    fig_dcm.tight_layout()

    dcm_plot_filename = f"analysis/v0=1.0_dcm.png"
    fig_dcm.savefig(dcm_plot_filename, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Gráfico de DCM guardado en: {dcm_plot_filename}")

    # --- PLOT 2: Error Function E(D) ---
    fig_error, ax_error = plt.subplots(figsize=(12, 8))

    ax_error.plot(
        d_range,
        fit_errors,
        color="#007ACC",
        linewidth=2.5,
        label="Error del ajuste E(D)",
    )
    # ax_error.set_title("Error del Ajuste en Función del Coeficiente de Difusión", fontsize=14, fontweight='bold')
    ax_error.set_xlabel("Coeficiente de Difusión D (m²/s)", fontsize=14)
    ax_error.set_ylabel("Error Cuadrático Total E(D)", fontsize=14)

    min_error_d = optimal_D
    min_error = np.min(fit_errors)
    ax_error.axvline(
        x=min_error_d,
        color="#D62728",
        linestyle="--",
        linewidth=2,
        label=f"D óptimo = {min_error_d:.2e} m²/s",
    )
    ax_error.plot(
        min_error_d,
        min_error,
        "o",
        color="#D62728",
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=1.5,
    )

    ax_error.legend(
        loc="upper right", fontsize=10, framealpha=0.9, fancybox=True, shadow=True
    )
    ax_error.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax_error.tick_params(axis="both", which="major", labelsize=12)
    ax_error.yaxis.set_major_formatter(FuncFormatter(format_func))
    fig_error.tight_layout()

    error_plot_filename = f"analysis/v0=1.0_dcm_error.png"
    fig_error.savefig(
        error_plot_filename, dpi=300, bbox_inches="tight", facecolor="white"
    )
    print(f"Gráfico de Error guardado en: {error_plot_filename}")

    # Show plots when run interactively
    # plt.show()

    # Print detailed analysis information
    print(f"\n=== ANÁLISIS DETALLADO ===")
    print(f"Partícula analizada: ID {args.particle_id} (M=3kg)")
    print(f"Velocidad inicial partículas pequeñas: v0 = 1 m/s")
    print(f"dt_bin para agrupación: {dt_bin:.6f} s")
    print(f"Coeficiente de difusión obtenido: D = {D:.6e} m²/s")
    print(f"Calidad del ajuste: R² = {r_squared:.6f}")
    print(f"Pendiente del ajuste: {slope:.6e} m²/s")
    print(f"Desviación estándar de residuos: {residual_std:.6e} m²")
    print(f"Residuo máximo absoluto: {np.max(np.abs(residuals)):.6e} m²")

    # Justify dt_bin choice based on timing analysis
    if intervals is not None and len(intervals) > 0:
        median_interval = np.median(intervals)
        ratio = dt_bin / median_interval
        print(f"\n=== JUSTIFICACIÓN DE dt_bin ===")
        print(f"dt_bin elegido: {dt_bin:.6f} s")
        print(f"Intervalo mediano entre eventos: {median_interval:.6f} s")
        print(f"Relación dt_bin/intervalo_mediano: {ratio:.1f}")
        print(
            f"Justificación: dt_bin es {ratio:.1f}x más grande que el intervalo típico"
        )
        print(f"entre eventos, lo que permite:")
        print(f"  • Suavizado estadístico al promediar múltiples eventos por bin")
        print(f"  • Reducción de ruido manteniendo resolución temporal adecuada")
        print(f"  • Equilibrio entre precisión temporal y estabilidad estadística")


if __name__ == "__main__":
    main()
