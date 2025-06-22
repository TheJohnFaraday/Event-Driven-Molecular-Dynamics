#!/bin/bash

# Simulation parameters
number_of_particles=202       # Default: 50
# radius=0.0005               # Default: 0.0005 m
# mass=1.0                    # Default: 1.0 kg
# initial_velocity=1.0          # Default: 1.0 m/s
final_time=10.0               # No default, required
output_directory=./output     # Required
seed=10000            # Optional, for reproducibility. MUST BE INT

rm output/*

# Clear the terminal
clear

# Build the project
gradle clean build


gradle run --no-build-cache --rerun-tasks --args="\
  --number-of-particles $number_of_particles \
  -t $final_time \
  --v0 1.0 \
  --no-fixed-obstacle \
  --seed 10000 \
  --output-directory $output_directory" 

gradle run --no-build-cache --rerun-tasks --args="\
  --number-of-particles $number_of_particles \
  -t $final_time \
  --v0 1.0 \
  --no-fixed-obstacle \
  --seed 10001 \
  --output-directory $output_directory"

gradle run --no-build-cache --rerun-tasks --args="\
  --number-of-particles $number_of_particles \
  -t $final_time \
  --v0 1.0 \
  --no-fixed-obstacle \
  --seed 10002 \
  --output-directory $output_directory"

gradle run --no-build-cache --rerun-tasks --args="\
  --number-of-particles $number_of_particles \
  -t $final_time \
  --v0 1.0 \
  --no-fixed-obstacle \
  --seed 10003 \
  --output-directory $output_directory"

gradle run --no-build-cache --rerun-tasks --args="\
  --number-of-particles $number_of_particles \
  -t $final_time \
  --v0 1.0 \
  --no-fixed-obstacle \
  --seed 10004 \
  --output-directory $output_directory"

gradle run --no-build-cache --rerun-tasks --args="\
  --number-of-particles $number_of_particles \
  -t $final_time \
  --v0 1.0 \
  --no-fixed-obstacle \
  --seed 10005 \
  --output-directory $output_directory"

gradle run --no-build-cache --rerun-tasks --args="\
  --number-of-particles $number_of_particles \
  -t $final_time \
  --v0 1.0 \
  --no-fixed-obstacle \
  --seed 10006 \
  --output-directory $output_directory"