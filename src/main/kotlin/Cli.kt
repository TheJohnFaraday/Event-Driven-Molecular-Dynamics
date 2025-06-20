package ar.edu.itba.ss

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.options.*
import com.github.ajalt.clikt.parameters.types.double
import com.github.ajalt.clikt.parameters.types.int
import com.github.ajalt.clikt.parameters.types.path
import io.github.oshai.kotlinlogging.KotlinLogging
import java.nio.file.Path

class Cli : CliktCommand() {

    private val logger = KotlinLogging.logger {}

    private val numberOfParticles: Int by option("-N", "--number-of-particles")
        .int()
        .default(50)
        .help("Total number of particles")
        .check("Must be greater than 0") { it > 0 }

    private val radius: Double by option("-r", "--radius")
        .double()
        .default(5e-4)
        .help("Radius of the particles [m]")
        .check("Must be greater than 0") { it > 0.0 }

    private val mass: Double by option("-m", "--mass")
        .double()
        .default(1.0)
        .help("Mass of the particles [kg])")
        .check("Must be greater than 0") { it > 0.0 }

    private val initialVelocity: Double by option("-v", "--v0", "--initial-velocity")
        .double()
        .default(1.0)
        .help("Initial velocity of the particles [m/s]")
        .check("Must be non-negative") { it >= 0.0 }

    private val finalTime: Double by option("-t", "--simulation-time")
        .double()
        .required()
        .help("Total simulation time")
        .check("Must be greater than 0") { it > 0.0 }

    private val enableInternalCollisions: Boolean by option("--enable-internal-collisions")
        .flag(default = true)
        .help("Whether internal particles can collide with each other")

    private val eventDensity: Int? by option()
        .int()
        .help("[Optional] How many events should be skipped from writing to output file.")

    private val obstacleRadius: Double by option()
        .double()
        .default(5e-3)
        .check("Obstacle must have a radius greater than zero") { it > 0.0 }

    private val obstacleMass: Double? by option()
        .double()
        .default(3.0)
        .check("Obstacle mass must be greater than zero or undefined if it doesn't have mass") { it > 0.0 }

    private val fixedObstacle: Boolean by option("--fixed-obstacle")
        .flag("--no-fixed-obstacle", default = true)
        .help("Whether the obstacle is fixed (true) or mobile (false)")

    private val containerRadius: Double by option()
        .double()
        .default(5e-2)
        .check("Container must have a radius greater than zero and greater than the obstacle") { it > 0.0 && it > obstacleRadius }

    private val seed: Int by option("-s", "--seed")
        .int()
        .default(67)
        .help("[Optional] Seed for the RND")
        .check("Seed must be greater or equal to 0.") { it > 0 }

    private val outputDirectory: Path by option().path(
        canBeFile = false,
        canBeDir = true,
        mustExist = true,
        mustBeReadable = true,
        mustBeWritable = true
    ).required().help("Path to the output directory.")

    override fun run() {
        logger.info { "Starting simulation with the following parameters:" }
        logger.info { "Number of particles: $numberOfParticles" }
        logger.info { "Particle radius: $radius [m]" }
        logger.info { "Particle mass: $mass [kg]" }
        logger.info { "Initial velocity: $initialVelocity [m/s]" }
        logger.info { "Final time: $finalTime [s]" }
        logger.info { "Enable internal collisions: $enableInternalCollisions" }
        logger.info { "Seed: $seed" }
        logger.info { "Event density: $eventDensity" }
        logger.info { "Output directory: $outputDirectory" }
        logger.info { "Container Radius: $containerRadius"}
        logger.info { "Obstacle radius: $obstacleRadius" }
        logger.info { "Obstacle Mass: ${obstacleMass?.toString() ?: "Not Defined"}" }
        logger.info { "Fixed obstacle: $fixedObstacle" }
        logger.info { "Event Density: ${eventDensity?.toString() ?: "All events recorded"}" }

        val fileName = buildString {
            append("particles=$numberOfParticles")
            append("_radius=$radius")
            append("_mass=$mass")
            append("_v0=$initialVelocity")
            append("_t=$finalTime")
            append("_internalCollisions=$enableInternalCollisions")
            append("_seed=$seed")
            append("_containerRadius=$containerRadius")
            append("_obstacleRadius=$obstacleRadius")
            append("_fixedObstacle=$fixedObstacle")
            if (obstacleMass != null) {
                append("_obstacleMass=$obstacleMass")
            }
            if (eventDensity != null) {
                append("_eventDensity=$eventDensity")
            }
        }.replace(".", "_").replace("=", "-") + ".csv"

        val outputCsv = outputDirectory.resolve(fileName).toFile()

        val settings = Settings(
            seed = seed,
            numberOfParticles = numberOfParticles,
            radius = radius,
            mass = mass,
            initialVelocity = initialVelocity,
            obstacleRadius = obstacleRadius,
            obstacleMass = obstacleMass,
            containerRadius = containerRadius,
            outputFile = outputCsv,
            finalTime = finalTime,
            internalCollisions = enableInternalCollisions,
            eventDensity = eventDensity,
            fixedObstacle = fixedObstacle
        )

        runSimulation(settings)
    }
}