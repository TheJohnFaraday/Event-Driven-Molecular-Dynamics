package ar.edu.itba.ss

class CollisionData(
    val time: Double,
    val particleA: Particle,
    val particleB: Particle?
) : Comparable<CollisionData> {

    private val countA: Int = particleA.collisions
    private val countB: Int = particleB?.collisions ?: -1

    fun isValid(): Boolean {
        if (particleA.collisions != countA) {
            return false
        }
        return particleB == null || particleB.collisions == countB
    }

    override fun compareTo(other: CollisionData): Int {
        return time.compareTo(other.time)
    }

    override fun toString(): String {
        return "CollisionData(time=$time, particleA=$particleA, particleB=$particleB)"
    }
}