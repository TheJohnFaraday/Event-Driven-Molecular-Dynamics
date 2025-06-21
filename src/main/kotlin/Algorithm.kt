package ar.edu.itba.ss

import kotlin.math.*

class ParticleMotor(
    private val particles: List<Particle>,
    private val containerR: Double
) {

    fun predict(p: Particle, now: Double): List<CollisionData> {
        val list = mutableListOf<CollisionData>()
        val A = p.vx * p.vx + p.vy * p.vy
        if (A > 0) {
            val dx = p.x
            val dy = p.y
            val B = 2 * (dx * p.vx + dy * p.vy)
            val C = dx * dx + dy * dy - (containerR - p.r).pow(2)
            val disc = B * B - 4 * A * C
            if (disc > 0) {
                var t = (-B + sqrt(disc)) / (2 * A)
                if (t <= 0) t = (-B - sqrt(disc)) / (2 * A)
                if (t > 1e-12) list.add(CollisionData(now + t, p, null))
            }
        }

        for (q in particles) {
            if (q == p) continue
            val dx = q.x - p.x
            val dy = q.y - p.y
            val dvx = q.vx - p.vx
            val dvy = q.vy - p.vy
            val dvdr = dx * dvx + dy * dvy
            if (dvdr >= 0) continue
            val dvdv = dvx * dvx + dvy * dvy
            val drdr = dx * dx + dy * dy
            val sigma = p.r + q.r
            val d = dvdr * dvdr - dvdv * (drdr - sigma * sigma)
            if (d < 0) continue
            val t = -(dvdr + sqrt(d)) / dvdv
            if (t > 1e-12) {
                list.add(CollisionData(now + t, p, q))
            }
        }

        return list
    }

    fun resolve(c: CollisionData) {
        val a = c.particleA
        val b = c.particleB
        if (b == null) {
            var nx = a.x
            var ny = a.y
            val norm = hypot(nx, ny)
            nx /= norm
            ny /= norm
            val vDotN = a.vx * nx + a.vy * ny
            a.vx -= 2 * vDotN * nx
            a.vy -= 2 * vDotN * ny
            a.collisions++
            return
        }

        if (a.isFixed() || b.isFixed()) {
            val particle = if (a.isFixed()) b else a
            val dx = particle.x
            val dy = particle.y
            val alpha = atan2(dy, dx)
            val (vfx, vfy) = computeCollisionVelocity(particle.vx, particle.vy, alpha, 1.0, 1.0)
            particle.vx = vfx
            particle.vy = vfy
        } else {
            impactNextTc(a, b)
        }

        a.collisions++
        b.collisions++
    }

    fun computeCollisionVelocity(vx: Double, vy: Double, alpha: Double, cn: Double, ct: Double): Pair<Double, Double> {
        val cosA = cos(alpha)
        val sinA = sin(alpha)
        val cos2 = cosA * cosA
        val sin2 = sinA * sinA
        val sincos = sinA * cosA

        val m11 = -cn * cos2 + ct * sin2
        val m12 = -(cn + ct) * sincos
        val m21 = m12
        val m22 = -cn * sin2 + ct * cos2

        val vfx = m11 * vx + m12 * vy
        val vfy = m21 * vx + m22 * vy

        return vfx to vfy
    }

    fun impactNextTc(p1: Particle, p2: Particle) {
        val dxr = p2.x - p1.x
        val dyr = p2.y - p1.y
        val dvx = p2.vx - p1.vx
        val dvy = p2.vy - p1.vy
        val sigma = p1.r + p2.r
        val drdv = dxr * dvx + dyr * dvy
        val J = (2 * p1.m * p2.m * drdv) / (sigma * (p1.m + p2.m))
        val Jx = (J * dxr) / sigma
        val Jy = (J * dyr) / sigma

        p1.vx += Jx / p1.m
        p1.vy += Jy / p1.m
        p2.vx -= Jx / p2.m
        p2.vy -= Jy / p2.m
    }
}
