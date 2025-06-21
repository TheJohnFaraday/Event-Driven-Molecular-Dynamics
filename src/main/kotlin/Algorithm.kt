package ar.edu.itba.ss

import kotlin.math.*

private data class CollisionVelocity(
    val vfx: Double,
    val vfy: Double
)

class Algorithm(
    private val containerR: Double
) {
    fun predictParticleCollision(
        particle: Particle,
        now: Double,
        particles: List<Particle>
    ): List<CollisionData> {
        val collisions = mutableListOf<CollisionData>()

        particleCollision(particle, now)?.let { collisions.add(it) }

        particles.forEach { q ->
            if (q == particle) {
                return@forEach
            }
            val dx = q.x - particle.x
            val dy = q.y - particle.y
            val dvx = q.vx - particle.vx
            val dvy = q.vy - particle.vy
            val dvdr = dx * dvx + dy * dvy
            if (dvdr >= 0) {
                // inf
                return@forEach
            }
            val dvdv = dvx * dvx + dvy * dvy
            val drdr = dx * dx + dy * dy
            val sigma2 = (particle.radius + q.radius) * (particle.radius + q.radius)
            val d = dvdr * dvdr - dvdv * (drdr - sigma2)
            if (d < 0) {
                // inf
                return@forEach
            }
            // otherwise
            val t_c = -(dvdr + sqrt(d)) / dvdv
            if (t_c > EPSILON) {
                collisions.add(CollisionData(now + t_c, particle, q))
            }
        }

        return collisions
    }

    fun resolveCollisions(collisionData: CollisionData) =
        if (collisionData.particleB == null) {
            singleParticleCollision(collisionData.particleA)
        } else {
            doubleParticleCollision(collisionData.particleA, collisionData.particleB)
        }


    private fun computeCollisionVelocity(
        vx: Double,
        vy: Double,
        alpha: Double,
        cn: Double,
        ct: Double
    ): CollisionVelocity {
        val cosA = cos(alpha)
        val sinA = sin(alpha)
        val cos2 = cosA * cosA
        val sin2 = sinA * sinA

        val m11 = -cn * cos2 + ct * sin2
        val m12 = -(cn + ct) * sinA * cosA
        val m21 = m12
        val m22 = -cn * sin2 + ct * cos2

        val vfx = m11 * vx + m12 * vy
        val vfy = m21 * vx + m22 * vy

        return CollisionVelocity(vfx = vfx, vfy = vfy)
    }

    private fun nextTcImpact(p1: Particle, p2: Particle) {
        val dxr = p2.x - p1.x
        val dyr = p2.y - p1.y
        val dvx = p2.vx - p1.vx
        val dvy = p2.vy - p1.vy
        val sigma = p1.radius + p2.radius
        val drdv = dxr * dvx + dyr * dvy
        val J = (2 * p1.mass * p2.mass * drdv) / (sigma * (p1.mass + p2.mass))
        val Jx = (J * dxr) / sigma
        val Jy = (J * dyr) / sigma

        p1.vx += Jx / p1.mass
        p1.vy += Jy / p1.mass
        p2.vx -= Jx / p2.mass
        p2.vy -= Jy / p2.mass
    }

    private fun particleCollision(particle: Particle, now: Double): CollisionData? {
        // x = ( -b +- sqrt(b^2 - 4ac) ) / 2a
        val x2y2 = particle.x * particle.x + particle.y * particle.y
        val a = particle.vx * particle.vx + particle.vy * particle.vy
        if (a <= 0) {
            return null
        }
        val b = 2 * (particle.x * particle.vx + particle.y * particle.vy)
        val c = x2y2 - (containerR - particle.radius).pow(2)
        val discriminant = b * b - 4 * a * c
        if (discriminant <= 0) {
            return null
        }
        var t = (-b + sqrt(discriminant)) / (2 * a)
        if (t <= 0) {
            t = (-b - sqrt(discriminant)) / (2 * a)
        }
        if (t <= EPSILON) {
            return null
        }
        return CollisionData(now + t, particle, null)
    }

    private fun singleParticleCollision(particle: Particle) {
        val (nx, ny) = particle.positionNorm()
        val vDotN = particle.vx * nx + particle.vy * ny
        particle.vx -= 2 * vDotN * nx
        particle.vy -= 2 * vDotN * ny
        particle.collisions++
    }

    private fun doubleParticleCollision(particleA: Particle, particleB: Particle) {
        if (!particleA.isFixed() && !particleB.isFixed()) {
            nextTcImpact(particleA, particleB)
        } else {
            val particle = if (particleA.isFixed()) particleB else particleA
            val dx = particle.x
            val dy = particle.y
            val alpha = atan2(dy, dx)
            val (vfx, vfy) = computeCollisionVelocity(particle.vx, particle.vy, alpha, 1.0, 1.0)
            particle.vx = vfx
            particle.vy = vfy
        }

        particleA.collisions++
        particleB.collisions++
    }


    companion object {
        private const val EPSILON = 1e-12
    }
}
