package ar.edu.itba.ss

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
    val pq = PriorityQueue<CollisionData>()

    var now = 0.0
    for (p in particles) {
        pq.addAll(motor.predict(p, now))
    }

    settings.outputFile.bufferedWriter().use { out ->
        dump(out, now, particles)
        while (now < settings.finalTime) {
            val c = pq.poll() ?: break
            if (!c.isValid()) continue
            val dt = c.time - now

            for (p in particles) {
                p.move(dt)
            }

            now = c.time
            motor.resolve(c)

            pq.addAll(motor.predict(c.a, now))
            c.b?.let { pq.addAll(motor.predict(it, now)) }

            dump(out, now, particles)
        }
    }

    println("Simulation finished. States written to ${settings.outputFile}")
}

private fun dump(out: Appendable, t: Double, ps: List<Particle>) {
    out.appendLine(String.format(Locale.US, "%.6f", t))
    for (p in ps) {
        out.appendLine(
            String.format(Locale.US, "%d %.6f %.6f %.6f %.6f", p.id, p.x, p.y, p.vx, p.vy)
        )
    }
    out.appendLine("---")
}
