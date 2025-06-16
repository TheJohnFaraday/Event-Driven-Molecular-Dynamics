package ar.edu.itba.ss

class CollisionData(
    val time: Double,
    val a: Particle,
    val b: Particle?
) : Comparable<CollisionData> {

    private val countA: Int = a.collisions
    private val countB: Int = b?.collisions ?: -1

    fun isValid(): Boolean {
        if (a.collisions != countA) return false
        return b == null || b.collisions == countB
    }

    override fun compareTo(other: CollisionData): Int {
        return time.compareTo(other.time)
    }

    override fun toString(): String {
        return "CollisionData(time=$time, a=$a, b=$b)"
    }
}