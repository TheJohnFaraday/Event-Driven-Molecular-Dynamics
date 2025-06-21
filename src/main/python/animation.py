import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse
import os

# Parameters
L = 0.1  # Diameter of outer ring
R = 0.005  # Radius of inner obstacle
particle_radius = 5e-4

# Argument parser
parser = argparse.ArgumentParser(
    description="Animate particle states in circular container."
)
parser.add_argument(
    "-f", "--file", type=str, required=True, help="Path to the states_...txt file"
)
parser.add_argument(
    "-o",
    "--fixed_obstacle",
    action="store_true",
    help="If present, the obstacle is fixed at the center. Otherwise, it's the particle with ID=0.",
)
args = parser.parse_args()
input_file = args.file
fixed_obstacle = args.fixed_obstacle

# Read states.txt and parse frames
with open(input_file) as f:
    raw_lines = f.read().split("---")

frames = []
obstacle_positions = []
time_stamps = []

for block in raw_lines:
    lines = block.strip().split("\n")
    if len(lines) < 2:
        continue
    time = float(lines[0])
    time_stamps.append(time)
    positions = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        idx, x, y, vx, vy = map(float, parts)
        if int(idx) == 0 and not fixed_obstacle:
            obstacle_positions.append((x, y))
        else:
            positions.append((x, y))
    frames.append(positions)

# Si el obstáculo está fijo, llenamos su posición como (0, 0) en todos los tiempos
if fixed_obstacle:
    obstacle_positions = [(0, 0)] * len(frames)

num_particles = len(frames[0])
colors = np.random.rand(num_particles, 3)

# Create figure
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.set_xlim(-L / 2 - 0.005, L / 2 + 0.005)
ax.set_ylim(-L / 2 - 0.005, L / 2 + 0.005)

# Draw container boundary
circle_recinto = plt.Circle((0, 0), L / 2, color="orange", fill=False, linewidth=2)
ax.add_artist(circle_recinto)

# Draw obstacle
obstacle_circle = plt.Circle((0, 0), R, color="gray")
ax.add_artist(obstacle_circle)

# Draw particles
particle_circles = []
for i in range(num_particles):
    c = plt.Circle((0, 0), particle_radius, color=colors[i])
    ax.add_artist(c)
    particle_circles.append(c)

# Text for displaying time
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=12, color="blue")

# Update function
def update(i):
    frame_data = frames[i]
    for j, (x, y) in enumerate(frame_data):
        particle_circles[j].center = (x, y)
    ox, oy = obstacle_positions[i]
    obstacle_circle.center = (ox, oy)
    time_text.set_text(f"t = {time_stamps[i]:.3f} s")
    return particle_circles + [obstacle_circle, time_text]

# Create animation
TOTAL_DURATION = 10  # seconds
interval = (TOTAL_DURATION * 1000) / len(frames)
ani = animation.FuncAnimation(
    fig, update, frames=len(frames), interval=interval, blit=True
)

# Save animation
print("Saving animation...")
os.makedirs("animations", exist_ok=True)
input_filename = os.path.basename(input_file).replace(".txt", "")
output_path = f"animations/{input_filename}.mp4"

ani.save(
    output_path,
    writer="ffmpeg",
    fps=len(frames) // TOTAL_DURATION,
    dpi=100,
    extra_args=["-crf", "27", "-preset", "veryfast"],
)
print(f"Animation saved successfully at: {output_path}")
