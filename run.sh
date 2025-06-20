#!/bin/bash

# Simulation parameters
number_of_particles=202       # Default: 50
# radius=0.0005               # Default: 0.0005 m
# mass=1.0                    # Default: 1.0 kg
# initial_velocity=1.0          # Default: 1.0 m/s
final_time=10.0               # No default, required
output_directory=./output     # Required
seed=10000            # Optional, for reproducibility. MUST BE INT

# Clear the terminal
clear

# Build the project
gradle clean build

# Run the simulation
taskset -c 0 \
gradle run --no-build-cache --rerun-tasks --args="\
  --number-of-particles $number_of_particles \
  -t $final_time \
  --v0 1.0 \
  --seed $seed \
  --output-directory $output_directory" &


taskset -c 1 \
gradle run --no-build-cache --rerun-tasks --args="\
  --number-of-particles $number_of_particles \
  -t $final_time \
  --v0 3.0 \
  --seed $seed \
  --output-directory $output_directory" &


taskset -c 2 \
gradle run --no-build-cache --rerun-tasks --args="\
  --number-of-particles $number_of_particles \
  -t $final_time \
  --v0 6.0 \
  --seed $seed \
  --output-directory $output_directory" &

taskset -c 3 \
gradle run --no-build-cache --rerun-tasks --args="\
  --number-of-particles $number_of_particles \
  -t $final_time \
  --v0 10.0 \
  --seed $seed \
  --output-directory $output_directory" &

taskset -c 4 \
gradle run --no-build-cache --rerun-tasks --args="\
  --number-of-particles $number_of_particles \
  -t $final_time \
  --v0 3.6 \
  --seed $seed \
  --output-directory $output_directory" &

taskset -c 5 \
gradle run --no-build-cache --rerun-tasks --args="\
  --number-of-particles $number_of_particles \
  -t $final_time \
  --v0 1.0 \
  --no-fixed-obstacle \
  --seed $seed \
  --output-directory $output_directory" &

wait

