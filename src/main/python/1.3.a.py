import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

# Constants from your simulation
L = 0.1  # Container diameter
R = 0.005  # Obstacle radius
particle_radius = 5e-4

parser = argparse.ArgumentParser(
    description="Analyze first-time collisions from multiple simulation files."
)
parser.add_argument(
    "-f",
    "--output_file",
    nargs="+",
    type=str,
    help="List of simulation output files to analyze",
)
parser.add_argument(
    "-v",
    "--velocities",
    nargs="+",
    type=float,
    help="Initial velocities corresponding to each file (optional, for labeling)",
)

args = parser.parse_args()


@dataclass
class CollisionAnalysis:
    """Results from analyzing obstacle collisions for both 1.3.a and 1.3.b"""

    # Time evaluation points for plotting
    t_eval: np.ndarray

    # 1.3.a: First-time collisions
    first_time_collisions_count: np.ndarray  # Cumulative count vs time
    first_time_t90: Optional[float]  # Time when 90% particles collided
    unique_particles_count: int  # Total particles that collided

    # 1.3.b: All collisions
    all_collisions_count: np.ndarray  # Cumulative count vs time
    collision_rate: Optional[float]  # Steady-state collision rate (collisions/s)
    total_collisions: int  # Total collision count

    # Simulation metadata
    total_time: float  # Total simulation time
    filename: str  # Source file


def extract_obstacle_collisions_streaming(filename: str, target_percentage: float = 0.9):
    """Extract obstacle collision data with streaming analysis - stops at target percentage"""
    obstacle_collisions = []
    first_collision_particles = set()
    total_particles = None
    target_unique_particles = None
    
    print(f"  Starting streaming analysis (target: {target_percentage*100:.0f}% particles)")
    
    with open(filename, "r") as f:
        current_block = ""
        
        for line_num, line in enumerate(f):
            line = line.strip()
            
            # Check for block separator or end of file
            if line == "---" or not line:
                if current_block.strip():
                    # Process the completed block
                    block_lines = current_block.strip().split("\n")
                    if len(block_lines) >= 2:
                        try:
                            time = float(block_lines[0])
                            
                            # Count total particles in first frame
                            if total_particles is None:
                                total_particles = len(block_lines) - 1  # Exclude time line
                                # Count only particles that can collide (exclude obstacle with ID=0)
                                colliding_particles = total_particles - 1  # Exclude obstacle
                                target_unique_particles = int(target_percentage * colliding_particles)
                                print(f"  Detected {total_particles} total particles ({colliding_particles} can collide), target: {target_unique_particles} unique collisions")
                            
                            # Process particle data
                            for particle_line in block_lines[1:]:
                                parts = particle_line.strip().split()
                                if len(parts) < 5:
                                    continue

                                particle_id, x, y, vx, vy = map(float, parts)

                                if particle_id > 0:  # Only for particles, not obstacle
                                    # Calculate radial distance and radial velocity
                                    r = np.sqrt(x**2 + y**2)
                                    v_radial = (x * vx + y * vy) / r if r > 0 else 0

                                    # Check obstacle collision (particle hits obstacle)
                                    obstacle_distance = r - (particle_radius + R)
                                    if obstacle_distance <= 1e-4 and v_radial < 0:
                                        obstacle_collisions.append([time, int(particle_id)])
                                        first_collision_particles.add(int(particle_id))
                                        
                                        # Check if we've reached our target
                                        if len(first_collision_particles) >= target_unique_particles:
                                            print(f"  Reached target at time {time:.3f}s after {line_num+1} lines")
                                            print(f"  Unique particles: {len(first_collision_particles)}, Total collisions: {len(obstacle_collisions)}")
                                            return (
                                                np.array(obstacle_collisions),
                                                len(first_collision_particles),
                                                total_particles
                                            )
                                            
                        except (ValueError, IndexError):
                            continue  # Skip malformed blocks
                    
                current_block = ""
            else:
                current_block += line + "\n"
    
    # Process final block if exists
    if current_block.strip():
        # Same processing logic as above
        block_lines = current_block.strip().split("\n")
        if len(block_lines) >= 2:
            try:
                time = float(block_lines[0])
                for particle_line in block_lines[1:]:
                    parts = particle_line.strip().split()
                    if len(parts) < 5:
                        continue
                    particle_id, x, y, vx, vy = map(float, parts)
                    if particle_id > 0:
                        r = np.sqrt(x**2 + y**2)
                        v_radial = (x * vx + y * vy) / r if r > 0 else 0
                        obstacle_distance = r - (particle_radius + R)
                        if obstacle_distance <= 1e-4 and v_radial < 0:
                            obstacle_collisions.append([time, int(particle_id)])
                            first_collision_particles.add(int(particle_id))
            except (ValueError, IndexError):
                pass
    
    print(f"  Finished file: {len(first_collision_particles)} unique particles, {len(obstacle_collisions)} total collisions")
    
    return (
        np.array(obstacle_collisions) if obstacle_collisions else np.array([]).reshape(0, 2),
        len(first_collision_particles),
        total_particles or 0
    )


def analyze_obstacle_collisions_from_streaming(
    obstacle_collisions: np.ndarray, unique_particles_count: int, filename: str
) -> Optional[CollisionAnalysis]:
    """Analyze collision data from streaming extraction (already filtered to t90%)"""
    if len(obstacle_collisions) == 0:
        return None

    tiempos = obstacle_collisions[:, 0]
    ids = obstacle_collisions[:, 1].astype(int)
    
    # ── 1.3.a: First-time collisions analysis ──
    primer_choque_por_particula = {}
    for t, pid in zip(tiempos, ids):
        if pid not in primer_choque_por_particula:
            primer_choque_por_particula[pid] = t

    if not primer_choque_por_particula:
        return None

    primeros_choques = np.array(sorted(primer_choque_por_particula.values()))

    # Calculate t90% - since we stopped reading when we reached the target number of particles,
    # the t90% is the time of the last (highest index) first collision we detected
    # This represents when exactly 90% of all possible particles have collided for the first time
    first_time_t90 = primeros_choques[-1]  # The time when the target was reached
    analysis_time = first_time_t90
    
    print(f"  t₉₀% calculated as: {first_time_t90:.6f} s (time when target particles reached)")
    print(f"  Unique particles found: {len(primeros_choques)} (should be ~90% of colliding particles)")
    print(f"  First 5 collision times: {primeros_choques[:5]}")
    print(f"  Last 5 collision times: {primeros_choques[-5:]}")
    
    # ── 1.3.b: All collisions analysis (already filtered data) ──
    todos_choques = np.sort(tiempos)
    total_collisions = len(todos_choques)

    # Create time series for plotting
    t_eval = np.linspace(0, analysis_time, 100)

    # 1.3.a: Cumulative unique particles that have collided
    first_time_collisions_count = np.array(
        [np.sum(primeros_choques <= t) for t in t_eval]
    )

    # 1.3.b: Cumulative total collisions
    all_collisions_count = np.array([np.sum(todos_choques <= t) for t in t_eval])

    # 1.3.b: Collision rate (collisions per second) 
    # Use period from 50% to 100% of analysis time for steady state rate
    tiempo_inicio_steady = 0.5 * analysis_time
    tiempo_fin_steady = analysis_time
    choques_steady = np.sum((todos_choques >= tiempo_inicio_steady) & 
                           (todos_choques <= tiempo_fin_steady))
    duracion_steady = tiempo_fin_steady - tiempo_inicio_steady
    collision_rate = choques_steady / duracion_steady if duracion_steady > 0 else None

    return CollisionAnalysis(
        t_eval=t_eval,
        first_time_collisions_count=first_time_collisions_count,
        first_time_t90=first_time_t90,
        unique_particles_count=unique_particles_count,
        all_collisions_count=all_collisions_count,
        collision_rate=collision_rate,
        total_collisions=total_collisions,
        total_time=analysis_time,
        filename=filename,
    )


def group_analyses_by_velocity(analyses: List[CollisionAnalysis], velocities: List[float]) -> List[CollisionAnalysis]:
    """Group analyses by velocity and compute mean values for repeated velocities"""
    from collections import defaultdict
    import scipy.interpolate
    
    # Group analyses by velocity
    velocity_groups = defaultdict(list)
    for analysis, velocity in zip(analyses, velocities):
        velocity_groups[velocity].append(analysis)
    
    averaged_analyses = []
    
    for velocity, group_analyses in velocity_groups.items():
        if len(group_analyses) == 1:
            # Single file for this velocity, use as-is
            averaged_analyses.append(group_analyses[0])
        else:
            # Multiple files for this velocity, compute averages
            print(f"  Averaging {len(group_analyses)} files for v₀ = {velocity} m/s")
            
            # Average scalar metrics
            t90_values = [a.first_time_t90 for a in group_analyses if a.first_time_t90 is not None]
            collision_rates = [a.collision_rate for a in group_analyses if a.collision_rate is not None]
            unique_counts = [a.unique_particles_count for a in group_analyses]
            total_collisions = [a.total_collisions for a in group_analyses]
            
            avg_t90 = np.mean(t90_values) if t90_values else None
            avg_collision_rate = np.mean(collision_rates) if collision_rates else None
            avg_unique_count = int(np.mean(unique_counts))
            avg_total_collisions = int(np.mean(total_collisions))
            
            # Create common time grid for averaging curves
            min_time = min(a.t_eval.min() for a in group_analyses)
            max_time = max(a.t_eval.max() for a in group_analyses)
            t_eval_common = np.linspace(min_time, max_time, 100)
            
            # Interpolate and average curves
            first_time_curves = []
            all_collisions_curves = []
            
            for analysis in group_analyses:
                # Interpolate to common grid
                f_first = scipy.interpolate.interp1d(
                    analysis.t_eval, analysis.first_time_collisions_count,
                    kind='linear', bounds_error=False, fill_value=(0, analysis.first_time_collisions_count[-1])
                )
                first_time_curves.append(f_first(t_eval_common))
                
                f_all = scipy.interpolate.interp1d(
                    analysis.t_eval, analysis.all_collisions_count,
                    kind='linear', bounds_error=False, fill_value=(0, analysis.all_collisions_count[-1])
                )
                all_collisions_curves.append(f_all(t_eval_common))
            
            # Compute averages
            avg_first_time_count = np.mean(first_time_curves, axis=0)
            avg_all_collisions_count = np.mean(all_collisions_curves, axis=0)
            
            # Create averaged analysis object
            avg_filename = f"averaged_{len(group_analyses)}_files_v{velocity}"
            avg_total_time = np.mean([a.total_time for a in group_analyses])
            
            averaged_analysis = CollisionAnalysis(
                t_eval=t_eval_common,
                first_time_collisions_count=avg_first_time_count,
                first_time_t90=avg_t90,
                unique_particles_count=avg_unique_count,
                all_collisions_count=avg_all_collisions_count,
                collision_rate=avg_collision_rate,
                total_collisions=avg_total_collisions,
                total_time=avg_total_time,
                filename=avg_filename
            )
            
            averaged_analyses.append(averaged_analysis)
    
    # Sort by velocity (extract from filename or use original velocity order)
    velocity_order = {v: i for i, v in enumerate(sorted(set(velocities)))}
    averaged_analyses.sort(key=lambda a: velocity_order.get(
        next(v for v, analyses in velocity_groups.items() 
             if any(a.filename == a.filename for a in analyses)), 0))
    
    return averaged_analyses


def plot_collision_analysis(files: List[str], velocities: Optional[List[float]] = None):
    """Plot both 1.3.a and 1.3.b analyses for all files"""
    viridis = cm.get_cmap("viridis")

    # Storage for temperature analysis
    t90_data = []  # For 1.3.a
    collision_rate_data = []  # For 1.3.b

    # Process all files
    raw_analyses = []
    for i, filename in enumerate(files):
        print(f"Processing {filename}...")

        # Extract collision data with streaming (stops at 90%)
        obstacle_collisions, unique_count, total_particles = extract_obstacle_collisions_streaming(filename)

        # Analyze collisions
        analysis = analyze_obstacle_collisions_from_streaming(obstacle_collisions, unique_count, filename)

        if analysis is None:
            print(f"No obstacle collisions found in {filename}")
            continue

        raw_analyses.append(analysis)

        # Print stats for individual file
        print(f"  Analysis time window: 0 - {analysis.total_time:.3f} s (up to t₉₀%)")
        print(f"  Total unique particles: {analysis.unique_particles_count}")
        print(f"  Total collisions (in window): {analysis.total_collisions}")
        print(f"  t₉₀%: {analysis.first_time_t90:.3f} s")
        if analysis.collision_rate:
            print(f"  Collision rate: {analysis.collision_rate:.1f} collisions/s")

    # Group analyses by velocity (average if multiple files per velocity)
    if velocities:
        print("\nGrouping analyses by velocity...")
        analyses = group_analyses_by_velocity(raw_analyses, velocities[:len(raw_analyses)])
        
        # Prepare temperature data from grouped analyses
        unique_velocities = []
        for analysis in analyses:
            # Extract velocity from analysis (either original or from averaged filename)
            if "averaged" in analysis.filename:
                # Extract velocity from averaged filename
                v_str = analysis.filename.split("_v")[1]
                velocity = float(v_str)
            else:
                # Find original velocity for this analysis
                for i, raw_analysis in enumerate(raw_analyses):
                    if raw_analysis.filename == analysis.filename and i < len(velocities):
                        velocity = velocities[i]
                        break
                else:
                    continue
            
            unique_velocities.append(velocity)
            temperature = (velocity**2) / (1.0**2)
            t90_data.append((temperature, analysis.first_time_t90))
            collision_rate_data.append((temperature, analysis.collision_rate))
    else:
        analyses = raw_analyses

    # ── Plot 1.3.a: First-time collisions ──
    plt.figure(figsize=(10, 6))
    for i, analysis in enumerate(analyses):
        color = viridis(i / max(len(analyses) - 1, 1))

        if velocities and i < len(unique_velocities):
            velocity = unique_velocities[i]
            # Check if this is an averaged result
            if "averaged" in analysis.filename:
                n_files = int(analysis.filename.split("_")[1])
                label = f"v₀ = {velocity} m/s (avg of {n_files})"
            else:
                label = f"v₀ = {velocity} m/s"
        else:
            label = f"Analysis {i+1}"

        plt.plot(
            analysis.t_eval,
            analysis.first_time_collisions_count,
            label=label,
            color=color,
        )

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Nro. de choques únicos")
    plt.title("1.3.a: Partículas que colisionan por primera vez (hasta t₉₀%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("analysis/1.3a_choques_unicos_multi.png", dpi=300)
    # plt.show()

    # ── Plot 1.3.b: All collisions ──
    plt.figure(figsize=(10, 6))
    for i, analysis in enumerate(analyses):
        color = viridis(i / max(len(analyses) - 1, 1))

        if velocities and i < len(unique_velocities):
            velocity = unique_velocities[i]
            # Check if this is an averaged result
            if "averaged" in analysis.filename:
                n_files = int(analysis.filename.split("_")[1])
                label = f"v₀ = {velocity} m/s (avg of {n_files})"
            else:
                label = f"v₀ = {velocity} m/s"
        else:
            label = f"Analysis {i+1}"

        plt.plot(
            analysis.t_eval, analysis.all_collisions_count, label=label, color=color
        )

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Nro. de choques totales")
    plt.title("1.3.b: Todas las colisiones (hasta t₉₀%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("analysis/1.3b_choques_totales_multi.png", dpi=300)
    # plt.show()

    # Plot temperature analyses if velocities were provided
    if velocities and t90_data and collision_rate_data:
        plot_temperature_analyses(t90_data, collision_rate_data)


def plot_temperature_analyses(t90_data: List[tuple], collision_rate_data: List[tuple]):
    """Plot both scalar observables vs temperature"""
    # Filter out None values and sort by temperature
    t90_data_clean = [(temp, t90) for temp, t90 in t90_data if t90 is not None]
    collision_rate_data_clean = [
        (temp, rate) for temp, rate in collision_rate_data if rate is not None
    ]

    t90_data_clean.sort(key=lambda x: x[0])
    collision_rate_data_clean.sort(key=lambda x: x[0])

    viridis = cm.get_cmap("viridis")

    # ── Plot 1.3.a: t90% vs temperature ──
    if t90_data_clean:
        temperatures_a, t90_values = zip(*t90_data_clean)

        plt.figure(figsize=(8, 5))
        plt.plot(temperatures_a, t90_values, "o-", color=viridis(0.3), markersize=8)
        plt.xlabel("Temperatura relativa [T/T₀]")
        plt.ylabel("t₉₀% [s]")
        plt.title("1.3.a: Tiempo para que 90% de partículas colisionen")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("analysis/1.3a_t90_vs_temperatura.png", dpi=300)
        # plt.show()

        print("\n1.3.a Temperature Analysis:")
        for temp, t90 in zip(temperatures_a, t90_values):
            print(f"  T/T₀ = {temp:.1f}: t₉₀% = {t90:.3f} s")

    # ── Plot 1.3.b: Collision rate vs temperature ──
    if collision_rate_data_clean:
        temperatures_b, collision_rates = zip(*collision_rate_data_clean)

        plt.figure(figsize=(8, 5))
        plt.plot(
            temperatures_b, collision_rates, "o-", color=viridis(0.7), markersize=8
        )
        plt.xlabel("Temperatura relativa [T/T₀]")
        plt.ylabel("Tasa de colisiones [s⁻¹]")
        plt.title("1.3.b: Tasa de colisiones en estado estacionario")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("analysis/1.3b_collision_rate_vs_temperatura.png", dpi=300)
        # plt.show()

        print("\n1.3.b Temperature Analysis:")
        for temp, rate in zip(temperatures_b, collision_rates):
            print(f"  T/T₀ = {temp:.1f}: Collision rate = {rate:.1f} s⁻¹")


if __name__ == "__main__":
    plot_collision_analysis(args.output_file, args.velocities)
