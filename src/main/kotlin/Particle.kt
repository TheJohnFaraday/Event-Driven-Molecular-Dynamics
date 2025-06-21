package ar.edu.itba.ss

import kotlin.math.hypot

data class Particle(
    val id: Int,
    var x: Double,
    var y: Double,
    var vx: Double,
    var vy: Double,
    val radius: Double,
    val mass: Double,
    var collisions: Int = 0
) {

    fun move(dt: Double) {
        x += vx * dt
        y += vy * dt
    }

    fun isFixed(): Boolean = mass.isInfinite()

    fun positionNorm(): Pair<Double, Double> {
        var nx = this.x
        var ny = this.y
        val norm = hypot(nx, ny)
        nx /= norm
        ny /= norm
        return nx to ny
    }
}
