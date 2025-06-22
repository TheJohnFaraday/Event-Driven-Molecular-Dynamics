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

# MAXIMUM TIME SETTING - Set this to limit analysis time range
# Set to a number (e.g., 5.0) to limit to that time, or None for no limit
MAXIMUM_TIME = 10.0


parser = argparse.ArgumentParser(
    description="Parse Kotlin output files and generate pressure plots with averaging support.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Single file
  python 1.1.py -f single_file.csv
  
  # Multiple files (will be averaged)
  python 1.1.py -f file1.csv file2.csv file3.csv
  
  # Multiple files with fixed obstacle option
  python 1.1.py -f file1.csv file2.csv file3.csv -o
""",
)
parser.add_argument(
    "-o",
    "--fixed_obstacle",
    action="store_true",
    help="If present, it will assume that the obstacle is fixed at the center, otherwise it will use the last particle index as the obstacle to run calculations",
)
parser.add_argument(
    "-f",
    "--output_file",
    nargs="+",  # Allow multiple files
    type=str,
    required=True,
    help="Output file(s) to analyze. Multiple files will be averaged.",
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

    # Apply maximum time filter if specified
    if MAXIMUM_TIME is not None:
        print(f"  Applying maximum time filter: {MAXIMUM_TIME} s")
        mask = times <= MAXIMUM_TIME
        times = times[mask]
        p_wall = p_wall[mask]
        p_obstacle = p_obstacle[mask]
        print(f"  Filtered data points: {len(times)} (up to {max(times):.1f} s)")

    return times.tolist(), p_wall.tolist(), p_obstacle.tolist()


def analyze_equilibrium(times, pressures, label=""):
    """Analyze if the system has reached equilibrium"""
    if len(times) < 10:
        return None, None

    # Take last 50% of data for equilibrium analysis
    mid_point = len(times) // 2
    eq_times = np.array(times[mid_point:])
    eq_pressures = np.array(pressures[mid_point:])

    if len(eq_pressures) == 0:
        return None, None

    # Calculate statistics
    mean_pressure = np.mean(eq_pressures)
    std_pressure = np.std(eq_pressures)
    cv = std_pressure / mean_pressure if mean_pressure > 0 else float("inf")

    # Check trend using linear regression
    if len(eq_times) > 1:
        slope, _ = np.polyfit(eq_times, eq_pressures, 1)
        relative_slope = (
            abs(slope) / mean_pressure if mean_pressure > 0 else float("inf")
        )
    else:
        relative_slope = float("inf")

    print(f"Equilibrium analysis - {label}:")
    print(f"  Mean pressure: {mean_pressure:.6f} Pa")
    print(f"  Std deviation: {std_pressure:.6f} Pa")
    print(f"  Coefficient of variation: {cv:.4f}")
    print(f"  Relative slope: {relative_slope:.6f} (should be < 0.01 for equilibrium)")
    print(f"  Equilibrium reached: {cv < 0.1 and relative_slope < 0.01}")
    print()

    return mean_pressure, std_pressure


def plot_pressures(times, p_wall, p_obstacle, num_files=1):
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


def process_multiple_files(file_list):
    """Process multiple files and average their pressure results"""
    print(f"Processing {len(file_list)} file(s)...")

    all_results = []
    for i, filename in enumerate(file_list):
        print(f"  Processing file {i+1}/{len(file_list)}: {filename}")
        times, p_wall, p_obstacle = read_states_and_calculate_pressure(filename)

        if times:  # Only add if we got valid results
            all_results.append((times, p_wall, p_obstacle))
        else:
            print(f"    Warning: No valid data found in {filename}")

    if not all_results:
        print("Error: No valid data found in any file!")
        return [], [], []

    if len(all_results) == 1:
        print("Single file processed.")
        return all_results[0]

    # Multiple files - need to average
    print(f"\nAveraging results from {len(all_results)} files...")

    # Find common time grid - use the shortest time range
    min_max_time = min(max(times) for times, _, _ in all_results)
    max_min_time = max(min(times) for times, _, _ in all_results)

    # Apply maximum time constraint if specified
    if MAXIMUM_TIME is not None:
        min_max_time = min(min_max_time, MAXIMUM_TIME)
        print(f"  Applying maximum time constraint: {MAXIMUM_TIME} s")

    if max_min_time >= min_max_time:
        print("Warning: Files have non-overlapping time ranges. Using intersection.")

    # Create common time grid
    common_times = np.arange(0, min_max_time + DT_FIXED, DT_FIXED)

    # Interpolate each file's results to common grid
    wall_pressures = []
    obstacle_pressures = []

    for times, p_wall, p_obstacle in all_results:
        # Convert to numpy arrays for interpolation
        times_np = np.array(times)
        p_wall_np = np.array(p_wall)
        p_obstacle_np = np.array(p_obstacle)

        # Interpolate to common grid
        p_wall_interp = np.interp(common_times, times_np, p_wall_np)
        p_obstacle_interp = np.interp(common_times, times_np, p_obstacle_np)

        wall_pressures.append(p_wall_interp)
        obstacle_pressures.append(p_obstacle_interp)

    # Calculate averages
    wall_pressures = np.array(wall_pressures)
    obstacle_pressures = np.array(obstacle_pressures)

    avg_p_wall = np.mean(wall_pressures, axis=0)
    avg_p_obstacle = np.mean(obstacle_pressures, axis=0)

    print(f"✓ Averaged {len(all_results)} files successfully")
    print(f"  Time range: 0 - {max(common_times):.1f} s")
    print(f"  Data points: {len(common_times)}")

    return common_times.tolist(), avg_p_wall.tolist(), avg_p_obstacle.tolist()


if __name__ == "__main__":
    # Show maximum time setting
    if MAXIMUM_TIME is not None:
        print(f"🕒 Maximum time limit set to: {MAXIMUM_TIME} s")
    else:
        print("🕒 No maximum time limit (analyzing full simulation)")

    # Process file(s) - single or multiple with averaging
    times, p_wall, p_obstacle = process_multiple_files(args.output_file)

    if times:  # Only plot if we have valid data
        # Analyze equilibrium
        analyze_equilibrium(times, p_wall, "Wall pressure")
        analyze_equilibrium(times, p_obstacle, "Obstacle pressure")

        plot_pressures(times, p_wall, p_obstacle, len(args.output_file))
    else:
        print("No data to plot!")
