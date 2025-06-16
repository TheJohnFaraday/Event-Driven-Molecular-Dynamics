package ar.edu.itba.ss

data class Particle(
    val id: Int,
    var x: Double,
    var y: Double,
    var vx: Double,
    var vy: Double,
    val r: Double,
    val m: Double,
    var collisions: Int = 0
) {

    fun move(dt: Double) {
        x += vx * dt
        y += vy * dt
    }

    fun isFixed(): Boolean = m.isInfinite()
}
