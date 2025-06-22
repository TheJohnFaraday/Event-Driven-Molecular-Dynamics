import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm
import glob
from collections import defaultdict
import argparse
import os

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
        exponent = int(np.floor(np.log10(abs(value))))
        base = value / 10**exponent
        if np.isclose(base, 1):
            return r"$10^{%d}$" % exponent
        else:
            return r"$%.1f \times 10^{%d}$" % (base, exponent)
    except:
        return ""

def main():
    parser = argparse.ArgumentParser(description="Análisis de DCM para partícula grande (M=3kg) - Event-Driven MD")
    parser.add_argument("--tmax", type=float, default=None, 
                       help="Tiempo máximo para el análisis (s)")
    parser.add_argument("--dt-bin", type=float, default=0.001,
                       help="Intervalo de tiempo para agrupación (s) - debe ser consistente con el cálculo DCM")
    parser.add_argument("--particle-id", type=int, default=0,
                       help="ID de la partícula grande a analizar (default: 0)")
    args = parser.parse_args()

    # Main processing
    dt_bin = args.dt_bin
    all_files = sorted(glob.glob("output/*"))

    dcms_by_time = defaultdict(list)
    max_times = []
    
    print(f"Analizando partícula ID: {args.particle_id} (M=3kg)")
    print(f"dt_bin para agrupación: {dt_bin:.6f} s")

    # First pass: compute minimum common time
    for file in all_files:
        times, _ = load_positions(file, args.particle_id)
        if len(times) > 0:
            max_times.append(times[-1])
        
    if not max_times:
        print("Error: No valid data found in any files")
        exit()

    t_min = min(max_times)
    t_max = args.tmax if args.tmax is not None else t_min
    
    print(f"Tiempo de corte (mínimo entre todas las corridas): {t_min:.4f} s")
    if args.tmax is not None:
        print(f"Tiempo máximo limitado a: {t_max:.4f} s")

    # Second pass: collect DCMs
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

        # Calculate DCM from TRUE initial position (t=0)
        # This is crucial for correct DCM calculation
        initial_pos = positions[0]  # Position at t=0
        dcm = np.sum((positions - initial_pos)**2, axis=1)

        # Group by time bins for ensemble averaging
        binned = group_by_time_bin(times, dcm, dt_bin=dt_bin)
        for t_bin, z2 in binned.items():
            dcms_by_time[t_bin].extend(z2 if isinstance(z2, list) else [z2])

    # Sort and compute stats
    sorted_times = sorted(dcms_by_time.keys())
    mean_dcm = [np.mean(dcms_by_time[t]) for t in sorted_times]
    std_dcm  = [np.std(dcms_by_time[t]) for t in sorted_times]

    print(f"Datos cargados y agrupados: {len(sorted_times)} elementos")
    print(f"Rango temporal: {sorted_times[0]:.4f} - {sorted_times[-1]:.4f} s")

    # Linear fit for diffusion coefficient
    coeffs = np.polyfit(sorted_times, mean_dcm, 1)
    fit_line = np.polyval(coeffs, sorted_times)
    D = coeffs[0] / 4  # D = slope/4 for 2D diffusion
    
    # Calculate R² and residuals for fit quality
    residuals = np.array(mean_dcm) - np.array(fit_line)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((mean_dcm - np.mean(mean_dcm))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1
    residual_std = np.std(residuals)

    # Error analysis for E(D) plot
    optimal_intercept = coeffs[1]
    optimal_D = D
    
    # Create a range of D values to test around the optimal one
    d_range = np.linspace(optimal_D * 0.5, optimal_D * 1.5, 200)

    # Calculate the error E(D) for each D value
    # E(D) = sum_i [y_i - f(x_i, D)]^2, where f(x_i, D) = (4*D)*x_i + intercept
    fit_errors = []
    for d_val in d_range:
        current_slope = 4 * d_val
        predicted_dcm = current_slope * np.array(sorted_times) + optimal_intercept
        error = np.sum((np.array(mean_dcm) - predicted_dcm)**2)
        fit_errors.append(error)
    
    print(f"Coeficiente de difusión: D = {D:.6e} m²/s")
    print(f"R² del ajuste: {r_squared:.6f}")

    # Set style for better looking plots
    plt.style.use('default')
    
    # --- PLOT 1: DCM vs Time ---
    fig_dcm, ax_dcm = plt.subplots(figsize=(12, 8))
    
    ax_dcm.fill_between(sorted_times,
                   np.array(mean_dcm) - np.array(std_dcm),
                   np.array(mean_dcm) + np.array(std_dcm),
                   color="#2E86AB", alpha=0.2, label="±1σ")
    
    ax_dcm.plot(sorted_times, mean_dcm, color="#2E86AB", linewidth=1.2, alpha=0.9, label="DCM (Ensemble)")
    
    ax_dcm.plot(sorted_times, fit_line, color="#A23B72", linewidth=3, 
            label=f"Ajuste Lineal (D={D:.2e} m²/s)")
    
    #title_text_dcm = f"Desplazamiento Cuadrático Medio - Partícula Grande (M=3kg)\n{len(all_files)} archivos, dt={dt_bin:.3f}s"
    #if args.tmax is not None:
    #    title_text_dcm += f"\nRango: 0.0 - {t_max:.2f} s"
    
    #ax_dcm.set_title(title_text_dcm, fontsize=16, fontweight='bold', pad=20)
    ax_dcm.set_xlabel("Tiempo (s)", fontsize=14)
    ax_dcm.set_ylabel("DCM (m²)", fontsize=14)
    
    ax_dcm.legend(loc="upper left", fontsize=12, framealpha=0.9, fancybox=True, shadow=True)
    ax_dcm.grid(True, linestyle="--", alpha=0.3, color='gray')
    ax_dcm.set_ylim(bottom=0)
    
    #stats_text_dcm = f"R² = {r_squared:.4f}\nN = {len(all_files)} archivos\nPuntos = {len(sorted_times)}\nD = {D:.2e} m²/s"
    #ax_dcm.text(0.02, 0.98, stats_text_dcm, transform=ax_dcm.transAxes,
    #        verticalalignment='top', fontsize=11,
    #        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax_dcm.tick_params(axis='both', which='major', labelsize=12)
    ax_dcm.yaxis.set_major_formatter(FuncFormatter(format_func))
    fig_dcm.tight_layout()
    
    dcm_plot_filename = f"analysis/v0=1.0_dcm.png"
    fig_dcm.savefig(dcm_plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Gráfico de DCM guardado en: {dcm_plot_filename}")

    # --- PLOT 2: Error Function E(D) ---
    fig_error, ax_error = plt.subplots(figsize=(12, 8))
    
    ax_error.plot(d_range, fit_errors, color="#007ACC", linewidth=2.5, label="Error del ajuste E(D)")
    #ax_error.set_title("Error del Ajuste en Función del Coeficiente de Difusión", fontsize=14, fontweight='bold')
    ax_error.set_xlabel("Coeficiente de Difusión D (m²/s)", fontsize=14)
    ax_error.set_ylabel("Error Cuadrático Total E(D)", fontsize=14)
    
    min_error_d = optimal_D
    min_error = np.min(fit_errors)
    ax_error.axvline(x=min_error_d, color="#D62728", linestyle='--', linewidth=2, label=f"D óptimo = {min_error_d:.2e} m²/s")
    ax_error.plot(min_error_d, min_error, 'o', color="#D62728", markersize=10, markeredgecolor='white', markeredgewidth=1.5)
    
    ax_error.legend(loc="upper right", fontsize=10, framealpha=0.9, fancybox=True, shadow=True)
    ax_error.grid(True, linestyle="--", alpha=0.3, color='gray')
    ax_error.tick_params(axis='both', which='major', labelsize=12)
    ax_error.yaxis.set_major_formatter(FuncFormatter(format_func))
    fig_error.tight_layout()

    error_plot_filename = f"analysis/v0=1.0_dcm_error.png"
    fig_error.savefig(error_plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Gráfico de Error guardado en: {error_plot_filename}")
    
    # Show plots when run interactively
    plt.show()
    
    # Print detailed analysis information
    print(f"\n=== ANÁLISIS DETALLADO ===")
    print(f"Partícula analizada: ID {args.particle_id} (M=3kg)")
    print(f"Velocidad inicial partículas pequeñas: v0 = 1 m/s")
    print(f"dt_bin para agrupación: {dt_bin:.6f} s")
    print(f"Coeficiente de difusión obtenido: D = {D:.6e} m²/s")
    print(f"Calidad del ajuste: R² = {r_squared:.6f}")
    print(f"Pendiente del ajuste: {coeffs[0]:.6e} m²/s")
    print(f"Intercepto: {coeffs[1]:.6e} m²")
    print(f"Desviación estándar de residuos: {residual_std:.6e} m²")
    print(f"Residuo máximo absoluto: {np.max(np.abs(residuals)):.6e} m²")

if __name__ == "__main__":
    main()
