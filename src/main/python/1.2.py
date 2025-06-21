import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List, Tuple

L = 0.1
R = 0.005
DT_FIXED = 0.1
MASS = 1.0
SKIP_TIME = 2.0
particle_radius = 5e-4


def extract_v0_from_filename(filename: str) -> float:
    match = re.search(r"v0-([0-9_\.]+)", filename)
    if match:
        return float(match.group(1).replace("_", ".").rstrip("."))
    raise ValueError(f"No se pudo extraer v0 del nombre: {filename}")


def read_states_and_calculate_pressure(
    filename: str,
) -> Tuple[List[float], List[float], List[float]]:
    with open(filename, "r") as f:
        content = f.read()

    blocks = content.split("---")
    collision_data = []

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        time = float(lines[0])

        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            particle_id, x, y, vx, vy = map(float, parts)

            r = np.sqrt(x**2 + y**2)
            v_radial = (x * vx + y * vy) / r if r > 0 else 0

            is_wall_collision = False
            is_obstacle_collision = False

            if particle_id > 0:
                wall_distance = (L / 2) - r - 5e-4
                if wall_distance <= 1e-4 and v_radial > 0:
                    is_wall_collision = True

                obstacle_distance = r - (R + 5e-4)
                if obstacle_distance <= 1e-4 and v_radial < 0:
                    is_obstacle_collision = True

                if is_wall_collision or is_obstacle_collision:
                    collision_data.append(
                        {
                            "time": time,
                            "particle_id": int(particle_id),
                            "v_radial": abs(v_radial),
                            "mass": 1.0,
                            "type": "WALL" if is_wall_collision else "OBSTACLE",
                        }
                    )

    if not collision_data:
        return [], [], []

    df = pd.DataFrame(collision_data)
    df["impulse"] = 2 * df["mass"] * df["v_radial"]
    df["time_bin"] = np.floor(df["time"] / DT_FIXED).astype(int)

    impulse_sums = (
        df.groupby(["time_bin", "type"])["impulse"].sum().unstack(fill_value=0.0)
    )

    perimeter_container = 2 * np.pi * ((L / 2) - particle_radius)
    times = impulse_sums.index * DT_FIXED
    p_wall = impulse_sums.get("WALL", 0) / (DT_FIXED * perimeter_container)

    return times.tolist(), p_wall.tolist(), impulse_sums.get("OBSTACLE", 0).tolist()


def calcular_temp_media(filename: str, desde_t: float) -> float:
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
            vx = float(parts[3])
            vy = float(parts[4])
            v2_total += vx**2 + vy**2
            count += 1

    return v2_total / count if count > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Graficar P promedio vs T relativa")
    parser.add_argument("-f", "--output_files", nargs="+", required=True)
    args = parser.parse_args()

    presiones = []
    temperaturas = []
    v0s = []

    for file in args.output_files:
        v0 = extract_v0_from_filename(file)
        times, p_wall, _ = read_states_and_calculate_pressure(file)
        T_media = calcular_temp_media(file, desde_t=SKIP_TIME)

        if len(p_wall) == 0 or T_media == 0:
            continue

        p_steady = np.mean([p for t, p in zip(times, p_wall) if t >= SKIP_TIME])

        presiones.append(p_steady)
        temperaturas.append(T_media)
        v0s.append(v0)

    temperaturas = np.array(temperaturas)
    temperaturas_rel = temperaturas / temperaturas[0]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(temperaturas_rel, presiones, "o-", color="darkorange")
    for v, T, P in zip(v0s, temperaturas_rel, presiones):
        plt.annotate(f"v0={v}", (T, P), fontsize=9)
    plt.xlabel("Temperatura relativa (T / T₀)")
    plt.ylabel("Presión promedio [Pa]")
    plt.title("Presión promedio vs Temperatura relativa")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("1.2_presion_vs_temperatura.png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    main()
