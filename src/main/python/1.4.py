import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os


def parse_file(filename: str):
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
            if int(idx) == 0:
                positions.append((x, y))
                break

    return np.array(times), np.array(positions)


def calculate_dcm(times: np.ndarray, positions: np.ndarray):
    origin = positions[0]
    displacements = positions - origin
    dcm = np.sum(displacements**2, axis=1)
    return dcm


def calculate_std_dcm(dcm: np.ndarray, window: int = 10):
    half_window = window // 2
    std_dcm = np.zeros_like(dcm)

    for i in range(len(dcm)):
        left = max(0, i - half_window)
        right = min(len(dcm), i + half_window + 1)
        std_dcm[i] = np.std(dcm[left:right])

    return std_dcm


def linear_fit(times: np.ndarray, dcm: np.ndarray, tmin=None, tmax=None):
    if tmin is not None and tmax is not None:
        indices = (times >= tmin) & (times <= tmax)
    else:
        indices = np.ones(len(times), dtype=bool)

    x = times[indices]
    y = dcm[indices]

    slope, intercept, r, _, _ = linregress(x, y)
    D = slope / 4
    return D, slope, intercept, r**2, x, y


def plot_dcm(times, dcm, dcm_std, D, slope, intercept, r2, t_fit, dcm_fit, filename):
    fig, ax = plt.subplots(figsize=(9, 6))

    # Gráfico con barras de error
    ax.errorbar(times, dcm, yerr=dcm_std, fmt='o-', color="steelblue", markersize=4,
                capsize=2, elinewidth=1, label="DCM")

    ax.plot(t_fit, slope * t_fit + intercept, "r-", linewidth=2, label="Ajuste Lineal")

    dcm_max = np.max(dcm)
    ax.axhline(dcm_max, linestyle="--", color="red", linewidth=1.5, label="Desplazamiento Máximo (m²)")

    ax.set_xlim(t_fit[0], t_fit[-1])

    ax.set_title("Desplazamiento Cuadrático Medio", fontsize=14)
    ax.set_xlabel("Tiempo (s)", fontsize=12)
    ax.set_ylabel("Desplazamiento Cuadrático Medio (m²)", fontsize=12)

    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    os.makedirs("analysis", exist_ok=True)
    out_path = f"analysis/dcm_vs_tim.png"
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Gráfico guardado en: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="Archivo de salida de la simulación")
    parser.add_argument("--tmin", type=float, default=None, help="Tiempo mínimo para ajuste")
    parser.add_argument("--tmax", type=float, default=None, help="Tiempo máximo para ajuste")
    args = parser.parse_args()

    times, positions = parse_file(args.file)
    dcm = calculate_dcm(times, positions)
    dcm_std = calculate_std_dcm(dcm)

    D, slope, intercept, r2, t_fit, dcm_fit = linear_fit(times, dcm, args.tmin, args.tmax)

    print(f"Coef. de difusión: D ≈ {D:.4e} m²/s (R² = {r2:.4f})")
    base_filename = os.path.splitext(os.path.basename(args.file))[0]

    plot_dcm(times, dcm, dcm_std, D, slope, intercept, r2, t_fit, dcm_fit, base_filename)


if __name__ == "__main__":
    main()

