#!/usr/bin/env bash

# python src/main/python/1.3.a.py \
#   -v 1.0 3.0 6.0 10.0 \
#   -f \
#   output/particles-202_radius-5_0E-4_mass-1_0_v0-1_0_t-10_0_internalCollisions-true_seed-10000_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
#   output/particles-202_radius-5_0E-4_mass-1_0_v0-3_0_t-10_0_internalCollisions-true_seed-10000_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
#   output/particles-202_radius-5_0E-4_mass-1_0_v0-6_0_t-10_0_internalCollisions-true_seed-10000_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
#   output/particles-202_radius-5_0E-4_mass-1_0_v0-10_0_t-10_0_internalCollisions-true_seed-10000_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv 



python src/main/python/1.3.a.py \
  --v1 \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-1_0_t-10_0_internalCollisions-true_seed-10000_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-1_0_t-5_0_internalCollisions-true_seed-10001_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-1_0_t-5_0_internalCollisions-true_seed-10010_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-1_0_t-5_0_internalCollisions-true_seed-10011_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-1_0_t-5_0_internalCollisions-true_seed-10100_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  --v3 \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-3_0_t-10_0_internalCollisions-true_seed-10000_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-3_0_t-3_0_internalCollisions-true_seed-10001_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-3_0_t-3_0_internalCollisions-true_seed-10010_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-3_0_t-3_0_internalCollisions-true_seed-10011_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-3_0_t-3_0_internalCollisions-true_seed-10100_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  --v6 \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-6_0_t-10_0_internalCollisions-true_seed-10000_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-6_0_t-3_0_internalCollisions-true_seed-10001_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-6_0_t-3_0_internalCollisions-true_seed-10010_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-6_0_t-3_0_internalCollisions-true_seed-10011_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-6_0_t-3_0_internalCollisions-true_seed-10100_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  --v10 \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-10_0_t-10_0_internalCollisions-true_seed-10000_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-10_0_t-3_0_internalCollisions-true_seed-10001_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-10_0_t-3_0_internalCollisions-true_seed-10010_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-10_0_t-3_0_internalCollisions-true_seed-10011_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv \
  output/particles-202_radius-5_0E-4_mass-1_0_v0-10_0_t-3_0_internalCollisions-true_seed-10100_containerRadius-0_05_obstacleRadius-0_005_fixedObstacle-true_obstacleMass-3_0.csv

