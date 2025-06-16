package ar.edu.itba.ss

import kotlin.math.*
import kotlin.random.Random

class ParticleGenerator {
    fun generate(
        N: Int,
        v0: Double,
        smallR: Double,
        smallM: Double,
        containerR: Double,
        obstacleR: Double,
        obstacleFixed: Boolean,
        seed: Int
    ): List<Particle> {
        val list = mutableListOf<Particle>()
        val rnd = Random(seed)
        val obsM = if (obstacleFixed) Double.POSITIVE_INFINITY else 3.0

        list.add(Particle(0, 0.0, 0.0, 0.0, 0.0, obstacleR, obsM))

        var id = 1
        while (list.size < N + 1) {
            val angle = rnd.nextDouble() * 2 * PI
            val EPSILON = 0.001
            val r = obstacleR + smallR + rnd.nextDouble() * (containerR - smallR - obstacleR - EPSILON)
            val x = r * cos(angle)
            val y = r * sin(angle)

            val ok = list.none { p ->
                hypot(p.x - x, p.y - y) < p.r + smallR
            }

            if (!ok) continue

            val theta = rnd.nextDouble() * 2 * PI
            val vx = v0 * cos(theta)
            val vy = v0 * sin(theta)

            list.add(Particle(id++, x, y, vx, vy, smallR, smallM))
        }

        return list
    }
}