#!/usr/bin/env bash

python src/main/python/1.2.py \
  -f \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-1_0_t-10_0_internalCollisions-true_seed-10000_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-3_0_t-10_0_internalCollisions-true_seed-10000_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-6_0_t-10_0_internalCollisions-true_seed-10000_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-10_0_t-10_0_internalCollisions-true_seed-10000_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv
