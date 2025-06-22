#!/bin/bash

# Simulation parameters
number_of_particles=202       # Default: 50
# radius=0.0005               # Default: 0.0005 m
# mass=1.0                    # Default: 1.0 kg
# initial_velocity=1.0          # Default: 1.0 m/s
final_time=5.0               # No default, required
final_time_2=3.0               # No default, required
output_directory=./output     # Required
# seed_1=10000            # Optional, for reproducibility. MUST BE INT
seed_2=10001            # Optional, for reproducibility. MUST BE INT
seed_3=10010            # Optional, for reproducibility. MUST BE INT
seed_4=10011            # Optional, for reproducibility. MUST BE INT
seed_5=10100            # Optional, for reproducibility. MUST BE INT

# Clear the terminal
clear

# Stop old gradle
gradle --stop
# Start daemon
gradle --daemon

# Build the project
gradle clean build

# Run the simulation

# seed_5
taskset -c 0 \
gradle run --no-build-cache --rerun-tasks --args="\
  --number-of-particles $number_of_particles \
  -t $final_time \
  --v0 1.0 \
  --seed $seed_5 \
  --output-directory $output_directory" &


taskset -c 1 \
gradle run --no-build-cache --rerun-tasks --args="\
  --number-of-particles $number_of_particles \
  -t $final_time_2 \
  --v0 3.0 \
  --seed $seed_5 \
  --output-directory $output_directory" &


taskset -c 2 \
gradle run --no-build-cache --rerun-tasks --args="\
  --number-of-particles $number_of_particles \
  -t $final_time_2 \
  --v0 6.0 \
  --seed $seed_5 \
  --output-directory $output_directory" &

taskset -c 3 \
gradle run --no-build-cache --rerun-tasks --args="\
  --number-of-particles $number_of_particles \
  -t $final_time_2 \
  --v0 10.0 \
  --seed $seed_5 \
  --output-directory $output_directory" &

wait 
