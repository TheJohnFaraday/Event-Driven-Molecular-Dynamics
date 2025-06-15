plugins {
    kotlin("jvm") version "2.1.10"
    id("application")
}

group = "ar.edu.itba.ss"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

application {
    mainClass = "ar.edu.itba.ss.MainKt"
}

dependencies {
    // Clickt for CLI parsing
    implementation("com.github.ajalt.clikt:clikt:5.0.1")
    // Support for rendering markdown in help messages
    implementation("com.github.ajalt.clikt:clikt-markdown:5.0.1")
    // Logging
    implementation("org.slf4j:slf4j-simple:2.0.3")
    implementation("io.github.oshai:kotlin-logging-jvm:7.0.3")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.1")

    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(21)
}