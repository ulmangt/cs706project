#!/bin/bash
java -Djava.library.path=JCuda-All-0.5.0RC-src/lib -cp .:src/main/resources:target/classes:lib/antlr-2.7.7.jar:lib/antlr-runtime-3.4.jar:lib/fastutil-6.4.1.jar:lib/glimpse-core-1.2.2.jar:lib/glimpse-core-examples-1.2.2.jar:lib/glimpse-util-1.2.2.jar:lib/gluegen-rt-1.0b06.jar:lib/gluegen-rt-native-all-1.0b06.jar:lib/guava-12.0.jar:lib/jogl-1.1.1a.jar:lib/jogl-native-all-1.1.1a.jar:lib/jsr305-1.3.9.jar:lib/miglayout-core-4.2.jar:lib/stringtemplate-3.2.1.jar:JCuda-All-0.5.0RC-src/java-lib/jcublas-0.5.0RC.jar:JCuda-All-0.5.0RC-src/java-lib/jcuda-0.5.0RC.jar:JCuda-All-0.5.0RC-src/java-lib/jcufft-0.5.0RC.jar:JCuda-All-0.5.0RC-src/java-lib/jcurand-0.5.0RC.jar:JCuda-All-0.5.0RC-src/java-lib/jcusparse-0.5.0RC.jar edu.gmu.ulman.histogram.HeatMapHistogramViewer
