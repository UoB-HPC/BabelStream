lazy val mainCls = Some("scalastream.App")

lazy val root = (project in file("."))
  .enablePlugins(NativeImagePlugin)
  .settings(
    scalaVersion := "3.0.0",
    version := "4.0",
    organization := "uk.ac.bristol.uob-hpc",
    organizationName := "University of Bristol",
    Compile / mainClass := mainCls,
    assembly / mainClass := mainCls,
    scalacOptions ~= filterConsoleScalacOptions,
    assembly / assemblyJarName := "scala-stream.jar",
    nativeImageOptions := Seq(
      "--no-fallback",
      "-H:ReflectionConfigurationFiles=../../reflect-config.json"
    ),
    nativeImageVersion := "21.1.0",
    (Global / excludeLintKeys) += nativeImageVersion,
    name := "scala-stream",
    libraryDependencies ++= Seq(
      // Lazy val implementation in Scala 3 triggers an exception in nativeImage, use 2_13 for arg parsing for now otherwise we can't get to the benchmarking part
      ("com.github.scopt" %% "scopt" % "4.0.1").cross(CrossVersion.for3Use2_13),
      // par also uses lazy val at some point, so it doesn't work in nativeImage
      "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.3",
      "net.openhft"             % "affinity"                   % "3.21ea1",
      "org.slf4j"               % "slf4j-simple"               % "1.7.30" // for affinity
    )
  )
