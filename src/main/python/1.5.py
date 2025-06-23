import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.cm as cm
import glob
from collections import defaultdict
import argparse
import os
import re
import pandas as pd

# --- Configuration ---
SKIP_TIME = 0.9  # Time to wait for system to stabilize before calculating temperature
TMAX_MAP = {
    1.0: 0.6,
    3.0: 0.5,
    6.0: 0.15,
    10.0: 0.13,
}

# --- Utility Functions ---

def format_func(value, _):
    """Format function for better number display in plots."""
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

# --- Data Loading and Parsing ---

def load_positions(filename: str, particle_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Parses a simulation file to get the times and positions for a specific particle."""
    with open(filename, "r") as f:
        content = f.read()

    blocks = content.split("---")
    times, positions = [], []

    for i, block in enumerate(blocks):
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        try:
            time = float(lines[0])
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) < 5: continue
                idx = int(float(parts[0]))
                if idx == particle_id:
                    times.append(time)
                    positions.append((float(parts[1]), float(parts[2])))
                    break
        except (ValueError, IndexError):
            print(f"  [Warning] Skipping malformed block in {os.path.basename(filename)}, block {i+1}")
            continue
    return np.array(times), np.array(positions)

def get_velocity_from_filename(filename: str) -> float | None:
    """Extracts velocity (v0) from the filename using regex."""
    match = re.search(r'v0-(\d+_\d+)', filename)
    if match:
        return float(match.group(1).replace('_', '.'))
    return None

def group_by_time_bin(times, values, dt_bin):
    binned = defaultdict(list)
    for t, v in zip(times, values):
        t_bin = round(t / dt_bin) * dt_bin
        binned[t_bin].append(v)
    return binned

def calculate_mean_temperature(filename: str, desde_t: float) -> float:
    with open(filename, "r") as f:
        content = f.read()

    blocks = content.split("---")
    v2_total = 0.0
    count = 0

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        time = float(lines[0])
        if time < desde_t:
            continue

        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # This is the line to add
            if int(float(parts[0])) != 0:
                vx = float(parts[3])
                vy = float(parts[4])
                v2_total += vx**2 + vy**2
                count += 1
    return v2_total / count

# --- Analysis Mode 1: DCM vs. Time ---

def run_dcm_vs_time_analysis(args, files_by_velocity):
    print("\n--- Running DCM vs. Time Analysis ---")
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use specific colors instead of viridis colormap for better consistency
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#3A0CA3']

    for i, (velocity, files) in enumerate(files_by_velocity.items()):
        print(f"\nProcesando v0 = {velocity:.1f} m/s...")

        t_max = TMAX_MAP.get(velocity)
        if t_max is None:
            print(f"  [Warning] No hay t_max definido para v0={velocity}. Saltando este grupo.")
            continue

        dcms_by_time = defaultdict(list)

        for file in files:
            times, positions = load_positions(file, args.particle_id)
            if len(times) < 2: continue
            mask = times <= t_max
            times, positions = times[mask], positions[mask]
            if len(times) < 2: continue

            initial_pos = positions[0]
            dcm = np.sum((positions - initial_pos)**2, axis=1)
            binned = group_by_time_bin(times, dcm, args.dt_bin)
            for t_bin, z2 in binned.items(): dcms_by_time[t_bin].extend(z2)

        if not dcms_by_time: continue

        sorted_times = np.array(sorted(dcms_by_time.keys()))
        mean_dcm = np.array([np.mean(dcms_by_time[t]) for t in sorted_times])
        std_dcm = np.array([np.std(dcms_by_time[t]) for t in sorted_times])

        # Linear fit forced through origin
        slope = np.sum(sorted_times * mean_dcm) / np.sum(sorted_times**2)
        D = slope / 4
        fit_line = slope * sorted_times

        color = colors[i % len(colors)]

        # Plot with enhanced styling
        label = f"v₀ = {velocity:.1f} m/s (D={D:.2e} m²/s)"
        
        # Add fill between for standard deviation
        ax.fill_between(
            sorted_times,
            mean_dcm - std_dcm,
            mean_dcm + std_dcm,
            color=color,
            alpha=0.2,
            label=f"±1σ v₀={velocity:.1f}"
        )
        
        # Plot mean DCM with enhanced styling
        ax.plot(sorted_times, mean_dcm, color=color, linewidth=1.2, alpha=0.9, label=label)
        
        # Plot fit line with enhanced styling
        ax.plot(sorted_times, fit_line, color=color, linestyle='--', linewidth=2.5)

    ax.set_title("Comparación de DCM para Diferentes Velocidades Iniciales", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Tiempo (s)", fontsize=14)
    ax.set_ylabel("DCM (m²)", fontsize=14)
    ax.legend(title="Velocidad Inicial", fontsize=12, framealpha=0.9, fancybox=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.3, color="gray")
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))
    fig.tight_layout()
    
    output_filename = "plots/dcm_vs_time.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nGráfico comparativo guardado en: {output_filename}")
    #plt.show()

# --- Analysis Mode 2: DCM vs. Temperature ---

def run_dcm_vs_temp_analysis(args, files_by_velocity):
    print("\n--- Running DCM vs. Temperature Analysis ---")
    plot_data = []
    for velocity, files in files_by_velocity.items():
        print(f"\nProcesando v0 = {velocity:.1f} m/s...")

        t_max = TMAX_MAP.get(velocity)
        if t_max is None:
            print(f"  [Warning] No hay t_max definido para v0={velocity}. Saltando este grupo.")
            continue

        group_temps = [calculate_mean_temperature(f, args.skip_time) for f in files]
        group_dcms_at_tmax = []
        for file in files:
            times, positions = load_positions(file, args.particle_id)
            if len(times) > 1:
                mask = times <= t_max
                times_trunc = times[mask]
                positions_trunc = positions[mask]

                if len(positions_trunc) > 1:
                    group_dcms_at_tmax.append(np.sum((positions_trunc[-1] - positions_trunc[0])**2))

        if group_temps and group_dcms_at_tmax:
            plot_data.append({
                'temp': np.mean(group_temps),
                'dcm': np.mean(group_dcms_at_tmax),
                'std': np.std(group_dcms_at_tmax),
                'v0': velocity
            })
            print(f"Temperatura media: {np.mean(group_temps):.4f} K")
            print(f"DCM promedio: {np.mean(group_dcms_at_tmax):.4f} m²")
            print(f"Desviación estándar: {np.std(group_dcms_at_tmax):.4f} m²")

    if not plot_data:
        print("\nError: No se pudieron procesar datos para generar el gráfico.")
        return

    df = pd.DataFrame(plot_data).sort_values('temp')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Enhanced errorbar styling
    ax.errorbar(df['temp'], df['dcm'], yerr=df['std'], fmt='o', 
                color='#007ACC', markersize=10, capsize=8, elinewidth=2.5, 
                capthick=2, markeredgecolor='white', markeredgewidth=1.5,
                label="DCM Promedio")
    
    # Enhanced annotations
    for _, row in df.iterrows():
        ax.annotate(f"v₀={row['v0']:.1f}", (row['temp'], row['dcm']),
                    textcoords="offset points", xytext=(0,15), ha='center',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_title("DCM del Obstáculo vs. Temperatura del Sistema", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Temperatura Media (proporcional a <v²>)", fontsize=14)
    ax.set_ylabel("DCM @ t_max (m²)", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3, color="gray")
    ax.legend(fontsize=12, framealpha=0.9, fancybox=True, shadow=True)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))
    fig.tight_layout()
    
    output_filename = "plots/dcm_vs_temp.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nGráfico guardado en: {output_filename}")
    #plt.show()

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Análisis de DCM para la partícula grande.")
    parser.add_argument("--particle_id", type=int, default=0,
                        help="ID de la partícula grande a analizar.")
    # Args for 'time' mode
    #parser.add_argument("--tmax", type=float, default=None,
    #                    help="[Modo Time] Tiempo máximo para el análisis (s).")
    parser.add_argument("--dt-bin", type=float, default=0.001,
                        help="[Modo Time] Intervalo de tiempo para agrupación (s).")
    # Args for 'temp' mode
    parser.add_argument("--skip_time", type=float, default=SKIP_TIME,
                        help="[Modo Temp] Tiempo a ignorar para el cálculo de la temperatura.")
    args = parser.parse_args()

    files_by_velocity = defaultdict(list)
    for file in sorted(glob.glob("output/*")):
        velocity = get_velocity_from_filename(file)
        if velocity is not None:
            files_by_velocity[velocity].append(file)

    if not files_by_velocity:
        print("Error: No se encontraron archivos con el formato de velocidad esperado (ej: 'v0-1_0').")
        return

    print("Archivos agrupados por velocidad:")
    for v, files in files_by_velocity.items():
        print(f"  v0 = {v:.1f} m/s: {len(files)} archivos")

    run_dcm_vs_time_analysis(args, files_by_velocity)
    run_dcm_vs_temp_analysis(args, files_by_velocity)

if __name__ == "__main__":
    main()
