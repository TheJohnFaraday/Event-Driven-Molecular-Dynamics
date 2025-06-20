import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mticker

L = 0.1  # Diameter of outer ring
R = 0.005  # Radius of inner obstacle
particle_radius = 5e-4
obstacle_mass = float("inf")

DT_FIXED = 0.1

# ------------------------------

CUSTOM_PALETTE = [
    "#508fbe",  # blue
    "#f37120",  # orange
    "#4baf4e",  # green
    "#f2cb31",  # yellow
    "#c178ce",  # purple
    "#cd4745",  # red
    "#9ef231",  # light green
    "#50beaa",  # green + blue
    "#8050be",  # violet
    "#cf1f51",  # magenta
]
BLACK = "#1a1a1a"
GREY = "#6f6f6f"
LIGHT_GREY = "#bfbfbf"

PLT_THEME = {
    "axes.prop_cycle": plt.cycler(color=CUSTOM_PALETTE),  # Set palette
    "axes.spines.top": False,  # Remove spine (frame)
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.edgecolor": BLACK,
    "axes.titleweight": "normal",  # Optional: ensure title weight is normal (not bold)
    "axes.titlelocation": "center",  # Center the title by default
    "axes.titlecolor": GREY,  # Set title color
    "axes.labelcolor": BLACK,  # Set labels color
    "axes.labelpad": 12,
    "xtick.bottom": False,  # Remove ticks on the X axis
    "xtick.labelcolor": BLACK,  # Set Y ticks color
    "xtick.color": GREY,  # Set Y label color
    "ytick.labelcolor": BLACK,  # Set Y ticks color
    "ytick.color": GREY,  # Set Y label color
    "savefig.dpi": 128,
    "legend.frameon": False,
    "legend.labelcolor": BLACK,
    "figure.titlesize": 16,  # Set suptitle size
    "font.size": 22,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
}
plt.style.use(PLT_THEME)
sns.set_palette(CUSTOM_PALETTE)
sns.set_style(PLT_THEME)

DPI = 100
FIGSIZE = (1920 / DPI, 1080 / DPI)


# ------------------------------


@dataclass
class SimulationParameters:
    r_container: float
    n_particles: int
    r_obstacle: float
    m_obstacle: float | None
    seed: int
    perimeter_container: float
    perimeter_obstacle: float


def format_power_of_10(x):
    if x == 0:
        return "0"

    # Get the base and exponent
    base, exp = f"{x:.2e}".split("e")
    base = float(base)
    exp = int(exp)

    # Check if the decimal part is zero (e.g. 1.00 → 1)
    if round(base % 1, 2) == 0:
        base_str = f"{int(base)}"
    else:
        base_str = f"{base:.2f}"

    return f"{base_str}x10^{exp}"


def y_fmt(x, pos):
    """Format number as power of 10"""
    return format_power_of_10(x)


def plot_pressure(pressure_df: pd.DataFrame, output_dir: str):
    plot_df = pressure_df.reset_index().rename(
        columns={"pressure_container": "recinto", "pressure_obstacle": "obstáculo"}
    )
    unified_plot_df = pressure_df.reset_index().melt(
        id_vars="time",
        value_vars=["pressure_container", "pressure_obstacle"],
        var_name="boundary",
        value_name="pressure_Pa",
    )

    # Map English to Spanish for the legend
    unified_plot_df["boundary"] = unified_plot_df["boundary"].map(
        {"pressure_container": "Recinto", "pressure_obstacle": "Obstáculo"}
    )

    # Unified plot
    plt.figure(figsize=FIGSIZE)
    sns.lineplot(
        data=unified_plot_df,
        x="time",
        y="pressure_Pa",
        hue="boundary",
        style="boundary",
    )
    plt.xlabel("Tiempo [s]")  # Changed to Spanish with units
    plt.ylabel("Presión [Pa]")  # Changed to Spanish with units
    plt.grid(True)
    plt.savefig(f"./{output_dir}/pressure.png")
    plt.clf()
    plt.close()

    # Container only
    plt.figure(figsize=FIGSIZE)
    sns.lineplot(data=plot_df, x="time", y="recinto")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Presión [Pa]")
    plt.grid(True)
    plt.savefig(f"./{output_dir}/pressure_container.png")
    plt.clf()
    plt.close()

    # Obstacle only
    plt.figure(figsize=FIGSIZE)
    sns.lineplot(data=plot_df, x="time", y="obstáculo")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Presión [Pa]")
    plt.grid(True)
    plt.savefig(f"./{output_dir}/pressure_obstacle.png")
    plt.clf()
    plt.close()


def plot_collisions_with_obstacle(
    simulation_params: SimulationParameters,
    df: pd.DataFrame,
    first_time_hits: pd.Series,
    all_hits: pd.Series,
    output_dir: str,
):
    unified_plot_df = pd.concat(
        [
            first_time_hits.rename("Primera vez"),
            all_hits.rename("Todas las colisiones"),
        ],
        axis=1,
    ).fillna(0)

    plt.figure(figsize=FIGSIZE)
    sns.lineplot(data=unified_plot_df, markers=True)
    plt.xlabel("Tiempo (s)", fontsize=14)
    plt.ylabel("Cantidad de colisiones", fontsize=14)
    plt.grid(True)
    # plt.show()
    plt.savefig(f"./{output_dir}/collissions_with_obstacle" ".png")
    plt.clf()
    plt.close()


def classify_collision(df: pd.DataFrame, sim: SimulationParameters, tol: float = 1e-5):
    """
    Returns a Series indexed exactly like `df` with values:
      'WALL' | 'OBSTACLE' | np.nan  (no collision)
    """
    # distance to outer wall (container)
    d_wall = sim.r_container - df["r"] - df["radius"]
    hit_wall = (df["v_n"] > 0) & (d_wall <= tol)

    # distance to inner obstacle
    d_obstacle = df["r"] - (df["radius"] + sim.r_obstacle)
    hit_obs = (df["v_n"] < 0) & (d_obstacle <= tol)

    out = pd.Series(index=df.index, dtype="object")
    out.loc[hit_wall] = "WALL"
    out.loc[hit_obs] = "OBSTACLE"
    return out


def calculate_pressure(sim_params: SimulationParameters, df: pd.DataFrame):
    # P = J / (delta t * L)
    coll = df.dropna(subset=["type"]).copy()

    # J = 2 m |v_n|
    coll["j"] = 2.0 * coll["m"] * coll["v_n"].abs()

    times = coll.index.get_level_values(0).to_numpy()
    coll["bin_id"] = np.floor_divide(times, DT_FIXED).astype(int)

    j_sum = (
        coll.groupby(["bin_id", "type"], sort=True)["j"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=["WALL", "OBSTACLE"], fill_value=0.0)
    )
    j_sum = j_sum.reindex(columns=["WALL", "OBSTACLE"], fill_value=0.0)

    P_cont = j_sum["WALL"] / (DT_FIXED * sim_params.perimeter_container)  # Pa
    P_obs = j_sum["OBSTACLE"] / (DT_FIXED * sim_params.perimeter_obstacle)  # Pa
    return (
        pd.DataFrame(
            {
                # "time": t,
                "time": j_sum.index * DT_FIXED,
                "pressure_container": P_cont,
                "pressure_obstacle": P_obs,
            }
        )
        .set_index("time")
        .sort_index()
        .iloc[:-1]
    )


def calculate_first_time_collisions(df: pd.DataFrame):
    # Keep only "OBSTACLE" collisions
    obs = df[df["type"] == "OBSTACLE"]
    if obs.empty:
        return pd.Series(dtype=int)

    # Keep first collision only. One row per id
    first_hit = obs.sort_values("time").groupby("id", sort=False).head(1)

    # time slices
    t = first_hit.index.get_level_values(0).to_numpy()
    bin_id = np.floor_divide(t, DT_FIXED).astype(int)
    first_hit = first_hit.copy()
    first_hit["bin_id"] = bin_id

    # count
    counts = first_hit.groupby("bin_id").size().rename("n_first_hits")
    counts.index = counts.index * DT_FIXED
    counts.index.name = "time"

    return counts


def calculate_all_obstacle_collisions(df: pd.DataFrame):
    # Keep only "OBSTACLE" collisions
    obs = df[df["type"] == "OBSTACLE"]
    if obs.empty:
        return pd.Series(dtype=int)

    # time slices
    t = obs.index.get_level_values(0).to_numpy()
    bin_id = np.floor_divide(t, DT_FIXED).astype(int)

    # count
    counts = (
        pd.Series(bin_id)
        .value_counts(sort=False)
        .rename_axis("bin_id")
        .sort_index()
        .rename("n_hits")
    )
    counts.index = counts.index * DT_FIXED
    counts.index.name = "time"

    return counts


def calculate_collisions_with_obstacle(
    sim_params: SimulationParameters, df: pd.DataFrame
):
    coll = df.dropna(subset=["type"]).copy()
    first_time = calculate_first_time_collisions(coll)
    print(first_time)

    all_hits = calculate_all_obstacle_collisions(coll)
    print(all_hits)

    return first_time, all_hits


def calculate_mqd(sim_params: SimulationParameters, df: pd.DataFrame):
    # Filter out the obstacle (id = 0) and only keep particles
    particle_df = df[df["id"] > 0].copy()

    # Set both time and id as index, then sort
    particle_df = particle_df.reset_index().set_index(["time", "id"]).sort_index()

    # Select only position columns
    particle_df = particle_df[["x", "y"]]

    # Unstack to get particles as columns
    xy_wide = particle_df.unstack(level=1)
    x = xy_wide["x"]
    y = xy_wide["y"]

    # Get initial positions (first time step)
    x0 = x.iloc[0]
    y0 = y.iloc[0]

    # Calculate mean squared displacement
    dr2 = (x.sub(x0, axis=1) ** 2) + (y.sub(y0, axis=1) ** 2)
    msd = dr2.mean(axis=1)

    msd.name = "msd"
    msd.index.name = "time"

    return msd


def read_states_file(filepath: str):
    """Read the states file generated by the Kotlin simulation"""
    with open(filepath, "r") as f:
        content = f.read()

    blocks = content.split("---")
    data_rows = []

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

            # Calculate radial distance and velocity
            r = np.sqrt(x**2 + y**2)
            v_radial = (x * vx + y * vy) / r if r > 0 else 0

            data_rows.append(
                {
                    "time": time,
                    "id": int(particle_id),
                    "x": x,
                    "y": y,
                    "vx": vx,
                    "vy": vy,
                    "r": r,
                    "v_n": v_radial,
                    "radius": (
                        particle_radius if particle_id > 0 else 0
                    ),  # obstacle has id=0
                    "m": (1.0 if particle_id > 0 else float("inf")),
                    "m_obstacle": obstacle_mass,
                }
            )

    df = pd.DataFrame(data_rows).set_index("time")

    # Create simulation parameters (you'll need to adjust these values)
    sim_params = SimulationParameters(
        r_container=L / 2,  # L = 0.1 from your animation.py
        n_particles=len(
            df[df["id"] > 0]["id"].unique()
        ),  # count non-obstacle particles
        r_obstacle=R,  # R = 0.005 from your animation.py
        m_obstacle=obstacle_mass,  # fixed obstacle
        seed=0,  # unknown
        perimeter_container=2 * np.pi * (L / 2),
        perimeter_obstacle=2 * np.pi * R,
    )

    return sim_params, df


def read_csv(filepath: str):
    config_df = pd.read_csv(filepath, nrows=1, header=0, keep_default_na=False)
    config = SimulationParameters(
        r_container=float(config_df["L/2"][0]),
        n_particles=int(config_df["n"][0]),
        r_obstacle=float(config_df["R"][0]),
        m_obstacle=(
            None if config_df["m_R"][0] == "null" else float(config_df["m_R"][0])
        ),
        seed=int(config_df["seed"][0]),
        perimeter_container=(2 * np.pi * config_df["L/2"][0]),
        perimeter_obstacle=(2 * np.pi * config_df["R"][0]),
    )

    df = pd.read_csv(
        filepath,
        sep=",",  # separator (default is comma)
        header=0,  # use first row as header
        index_col=None,  # don't use any column as index
        skiprows=2,  # number of rows to skip
    )
    df.set_index("time", inplace=True)

    return config, df


def main(output_file: str, fixed_obstacle: bool):
    output_base_dir = "./analysis"
    os.makedirs(output_base_dir, exist_ok=True)

    simulation_params, df = read_states_file(f"{output_file}")
    print(df)

    df["type"] = classify_collision(df, simulation_params)

    pressure_df = calculate_pressure(simulation_params, df)
    plot_pressure(pressure_df, output_base_dir)
    # print(pressure_df.head())

    # first_time_hits, all_hits = calculate_collisions_with_obstacle(simulation_params, df)
    # plot_collisions_with_obstacle(simulation_params, df, first_time_hits, all_hits, output_dir=output_base_dir)

    msd = calculate_mqd(simulation_params, df)
    print(msd)


if __name__ == "__main__":
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

    main(
        output_file=args.output_file,
        fixed_obstacle=True if args.fixed_obstacle else False,
    )
