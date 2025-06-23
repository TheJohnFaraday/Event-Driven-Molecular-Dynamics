#!/bin/bash

# Simulation parameters
number_of_particles=202       # Default: 50
# radius=0.0005               # Default: 0.0005 m
# mass=1.0                    # Default: 1.0 kg
# initial_velocity=1.0          # Default: 1.0 m/s
final_time=0.9               # No default, required
output_directory=./output     # Required
seed=10000            # Optional, for reproducibility. MUST BE INT

rm output/*

# Clear the terminal
clear

# Build the project
gradle clean build


# Define arrays for velocities and seeds
velocities=(1.0 3.0 6.0 10.0)
seeds=(10000 10001 10002 10003 10004 10005 10006 10007 10008 10009 10010 10011 10012 10013 10014 10015 10016 10017 10018 10019)

# Loop through velocities
for v0 in "${velocities[@]}"; do
  # Loop through seeds for each velocity
  for seed in "${seeds[@]}"; do
    gradle run --no-build-cache --rerun-tasks --args="\
      --number-of-particles $number_of_particles \
      -t $final_time \
      --v0 $v0 \
      --no-fixed-obstacle \
      --seed $seed \
      --output-directory $output_directory"
  done
done
