#!/bin/bash

# Simulation parameters
# number_of_particles=250     # Default: 250
# radius=0.0005               # Default: 0.0005 m
# mass=1.0                    # Default: 1.0 kg
# initial_velocity=1.0        # Default: 1.0 m/s
final_time=5.0               # No default, required
output_directory=./output     # Required
seed=1743645648280            # Optional, for reproducibility

# Clear the terminal
clear

# Build the project
gradle clean build

# Run the simulation
gradle run --no-build-cache --rerun-tasks --args="\
  -t $final_time \
  --output-directory $output_directory \
  -s $seed"
