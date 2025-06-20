import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Constants from your simulation
L = 0.1  # Container diameter
R = 0.005  # Obstacle radius
particle_radius = 5e-4

parser = argparse.ArgumentParser(
    description="Analyze first-time collisions from multiple simulation files."
)
parser.add_argument(
    "files", nargs="+", help="List of simulation output files to analyze"
)
parser.add_argument(
    "-v",
    "--velocities",
    nargs="+",
    type=float,
    help="Initial velocities corresponding to each file (optional, for labeling)",
)

args = parser.parse_args()


def extract_obstacle_collisions(filename: str):
    """Extract obstacle collision data from simulation file"""
    with open(filename, "r") as f:
        content = f.read()

    blocks = content.split("---")
    obstacle_collisions = []

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

            if particle_id > 0:  # Only for particles, not obstacle
                # Calculate radial distance and radial velocity
                r = np.sqrt(x**2 + y**2)
                v_radial = (x * vx + y * vy) / r if r > 0 else 0

                # Check obstacle collision (particle hits obstacle)
                obstacle_distance = r - (particle_radius + R)
                if obstacle_distance <= 1e-5 and v_radial < 0:
                    obstacle_collisions.append([time, int(particle_id)])

    return (
        np.array(obstacle_collisions)
        if obstacle_collisions
        else np.array([]).reshape(0, 2)
    )


def analyze_first_time_collisions(obstacle_collisions):
    """Analyze first-time collisions and return time series data"""
    if len(obstacle_collisions) == 0:
        return [], [], None

    tiempos = obstacle_collisions[:, 0]
    ids = obstacle_collisions[:, 1].astype(int)

    # Find first collision time for each particle
    primer_choque_por_particula = {}
    for t, pid in zip(tiempos, ids):
        if pid not in primer_choque_por_particula:
            primer_choque_por_particula[pid] = t

    if not primer_choque_por_particula:
        return [], [], None

    primeros_choques = np.array(sorted(primer_choque_por_particula.values()))

    # Create time series for plotting
    t_eval = np.linspace(0, tiempos.max(), 100)
    choques_unicos = [np.sum(primeros_choques <= t) for t in t_eval]

    # Calculate t90% as scalar observable
    N_total = len(primer_choque_por_particula)
    t_90 = primeros_choques[int(0.9 * N_total)] if N_total > 0 else None

    return t_eval, choques_unicos, t_90


def plot_all_collisions(files, velocities=None):
    """Plot first-time collisions for all files on the same graph"""
    plt.figure(figsize=(10, 6))
    viridis = cm.get_cmap("viridis")

    t90_data = []  # For temperature analysis

    for i, filename in enumerate(files):
        print(f"Processing {filename}...")

        # Extract collision data
        obstacle_collisions = extract_obstacle_collisions(filename)

        # Analyze first-time collisions
        t_eval, choques_unicos, t_90 = analyze_first_time_collisions(
            obstacle_collisions
        )

        if len(t_eval) == 0:
            print(f"No obstacle collisions found in {filename}")
            continue

        # Determine label and color
        if velocities and i < len(velocities):
            v0 = velocities[i]
            label = f"v₀ = {v0} m/s"
            t90_data.append((v0**2, t_90))  # Store (temperature, t90)
        else:
            label = f"File {i+1}"

        color = viridis(i / max(len(files) - 1, 1))

        # Plot
        plt.plot(t_eval, choques_unicos, label=label, color=color)

        print(
            f"  Total unique collisions: {max(choques_unicos) if choques_unicos else 0}"
        )
        if t_90:
            print(f"  t₉₀%: {t_90:.3f} s")

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Nro. de choques únicos")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("1.3a_choques_unicos_multi.png", dpi=300)
    # plt.show()

    # Plot temperature analysis if velocities were provided
    if velocities and t90_data:
        plot_temperature_analysis(t90_data)


def plot_temperature_analysis(t90_data):
    """Plot t90% vs temperature"""
    # Sort by temperature
    t90_data.sort(key=lambda x: x[0])
    temperatures, t90_values = zip(*t90_data)

    plt.figure(figsize=(8, 5))
    viridis = cm.get_cmap("viridis")
    plt.plot(temperatures, t90_values, "o-", color=viridis(0.6), markersize=8)
    plt.xlabel("Temperatura [m²·s⁻²]")
    plt.ylabel("t₉₀% [s]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("1.3a_t90_vs_temperatura.png", dpi=300)
    # plt.show()

    print("\nTemperature Analysis:")
    for temp, t90 in zip(temperatures, t90_values):
        print(f"  T = {temp:.1f} m²·s⁻²: t₉₀% = {t90:.3f} s")


if __name__ == "__main__":
    plot_all_collisions(args.files, args.velocities)
