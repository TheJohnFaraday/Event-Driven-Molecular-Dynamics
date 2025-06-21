import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

# Constants from your simulation
L = 0.1  # Container diameter
R = 0.005  # Obstacle radius
particle_radius = 5e-4
DT_FIXED = 0.1  # Time bin for pressure calculation

parser = argparse.ArgumentParser(
    description="Parse Kotlin output file and generate animations and plots."
)
parser.add_argument(
    "-o",
    "--fixed_obstacle",
    action="store_true",
    help="If present, it will assume that the obstacle is fixed at the center, otherwise it will use the last particle index as the obstacle to run calculations",
)
parser.add_argument(
    "-f", "--output_file", type=str, required=True, help="Output file to animate"
)

args = parser.parse_args()


def _pow10_fmt(y, _):
    """Return labels like 1.5×10^4 for scientific notation on axis."""
    if y == 0:
        return "0"
    exp = int(np.log10(abs(y)))
    mantissa = y / (10**exp)

    # # If mantissa is close to 1, just show 10^exp
    # if abs(mantissa - 1.0) < 0.05:
    #     if exp == 0:
    #         return "1"
    #     return rf"$10^{{{exp}}}$"

    # Otherwise show mantissa × 10^exp
    if exp == 0:
        return f"{mantissa:.1f}"
    return rf"${mantissa:.1f} \times 10^{{{exp}}}$"


def read_states_and_calculate_pressure(filename: str):
    """Read particle states and calculate pressure over time"""
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

            # Calculate radial distance and radial velocity
            r = np.sqrt(x**2 + y**2)
            v_radial = (x * vx + y * vy) / r if r > 0 else 0

            # Determine if this is a collision
            is_wall_collision = False
            is_obstacle_collision = False

            if particle_id > 0:  # Only for particles, not obstacle
                # Check wall collision (particle reaches container boundary)
                wall_distance = (L / 2) - r - particle_radius
                if wall_distance <= 1e-4 and v_radial > 0:
                    is_wall_collision = True

                # Check obstacle collision (particle hits obstacle)
                obstacle_distance = r - (particle_radius + R)
                if obstacle_distance <= 1e-4 and v_radial < 0:
                    is_obstacle_collision = True

                if is_wall_collision or is_obstacle_collision:
                    collision_data.append(
                        {
                            "time": time,
                            "particle_id": int(particle_id),
                            "v_radial": abs(v_radial),
                            "mass": 1.0,  # particle mass
                            "type": "WALL" if is_wall_collision else "OBSTACLE",
                        }
                    )

    if not collision_data:
        return [], [], []

    # Convert to DataFrame for easier processing
    df = pd.DataFrame(collision_data)

    # Calculate impulse J = 2 * m * |v_n|
    df["impulse"] = 2.0 * df["mass"] * df["v_radial"]

    # Group by time bins
    df["time_bin"] = np.floor(df["time"] / DT_FIXED).astype(int)

    # Sum impulses by time bin and collision type
    impulse_sums = (
        df.groupby(["time_bin", "type"])["impulse"].sum().unstack(fill_value=0.0)
    )

    # Calculate pressure: P = J / (dt * perimeter)
    perimeter_container = 2 * np.pi * ((L / 2) - particle_radius)
    perimeter_obstacle = 2 * np.pi * (R + particle_radius)

    # perimeter_container = 2 * np.pi * (L / 2)
    # perimeter_obstacle = 2 * np.pi * R

    times = impulse_sums.index * DT_FIXED
    p_wall = impulse_sums.get("WALL", 0) / (DT_FIXED * perimeter_container)
    p_obstacle = impulse_sums.get("OBSTACLE", 0) / (DT_FIXED * perimeter_obstacle)

    return times.tolist(), p_wall.tolist(), p_obstacle.tolist()


def plot_pressures(times, p_wall, p_obstacle):
    plt.figure(figsize=(10, 6))
    viridis = cm.get_cmap("viridis")

    ax = plt.gca()
    plt.plot(times, p_wall, label="P sobre recinto", color=viridis(0.2))
    plt.plot(times, p_obstacle, label="P sobre obstáculo", color=viridis(0.7))

    # Apply pretty scientific notation formatting to Y axis
    ax.yaxis.set_major_formatter(FuncFormatter(_pow10_fmt))

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Presión [Pa]")
    plt.legend()
    plt.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig("analysis/1.1_presion_vs_tiempo.png", dpi=300)
    # plt.show()


if __name__ == "__main__":
    times, p_wall, p_obstacle = read_states_and_calculate_pressure(args.output_file)
    plot_pressures(times, p_wall, p_obstacle)
