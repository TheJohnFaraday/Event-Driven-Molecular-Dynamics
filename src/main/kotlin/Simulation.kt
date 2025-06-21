package ar.edu.itba.ss

import java.io.BufferedWriter
import java.util.*

fun runSimulation(
    settings: Settings
) {
    val generator = ParticleGenerator()
    val particles = generator.generate(
        settings.numberOfParticles, settings.initialVelocity, settings.radius, settings.mass,
        settings.containerRadius, settings.obstacleRadius, settings.fixedObstacle, settings.seed
    )

    val motor = ParticleMotor(particles, settings.containerRadius)
    val priorityQueue = PriorityQueue<CollisionData>()

    var now = 0.0
    particles.forEach { particle -> priorityQueue.addAll(motor.predict(particle, now)) }

    settings.outputFile.bufferedWriter().use { out ->
        write(out, now, particles)
        while (now < settings.finalTime) {
            val collisionData = priorityQueue.poll() ?: break
            if (!collisionData.isValid()) continue
            val dt = collisionData.time - now

            particles.forEach { it.move(dt) }

            now = collisionData.time
            motor.resolve(collisionData)

            priorityQueue.addAll(motor.predict(collisionData.a, now))
            collisionData.b?.let { priorityQueue.addAll(motor.predict(it, now)) }

            write(out, now, particles)
        }
    }

    println("Simulation finished. States written to ${settings.outputFile}")
}

private fun write(out: BufferedWriter, time: Double, particles: List<Particle>) {
    out.appendLine("%.6f".format(time))
    particles.map { p ->
        listOf(
            p.id,
            "%.6f".format(p.x),
            "%.6f".format(p.y),
            "%.6f".format(p.vx),
            "%.6f".format(p.vy),
        ).joinToString(separator = " ")
    }.map { out.appendLine(it) }
    out.appendLine("---")
}
