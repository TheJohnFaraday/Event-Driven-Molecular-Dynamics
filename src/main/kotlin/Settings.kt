package ar.edu.itba.ss

import java.io.File

data class Settings(
    val numberOfParticles: Int,
    val radius: Double,
    val mass: Double,
    val initialVelocity: Double,
    val seed: Int,
    val obstacleRadius: Double,
    val obstacleMass: Double?,
    val containerRadius: Double,
    val outputFile: File,
    val finalTime: Double,
    val internalCollisions: Boolean,
    val fixedObstacle: Boolean,
    val eventDensity: Int?
)
