lazy val mainCls = Some("scalastream.App")

lazy val root = (project in file("."))
  .enablePlugins(NativeImagePlugin)
  .settings(
    scalaVersion := "3.3.1",
    version := "5.0",
    organization := "uk.ac.bristol.uob-hpc",
    organizationName := "University of Bristol",
    Compile / mainClass := mainCls,
    assembly / mainClass := mainCls,
    scalacOptions ~= filterConsoleScalacOptions,
    assembly / assemblyJarName := "scala-stream.jar",
    assembly / assemblyMergeStrategy := {
      case PathList("module-info.class")                                 => MergeStrategy.discard
      case PathList("META-INF", "versions", xs @ _, "module-info.class") => MergeStrategy.discard
      case x                                                             => (ThisBuild / assemblyMergeStrategy).value(x)
    },
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
      "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4",
      "net.openhft"             % "affinity"                   % "3.23.2",
      "org.slf4j"               % "slf4j-simple"               % "2.0.5" // for affinity
    )
  )
