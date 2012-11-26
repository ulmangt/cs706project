cs706project
============

JCUDA histogram calculation with Glimpse data visualization.

Compilation
============

Maven can be used to automatically download dependency jars and compile this project.

CUDA also requires that the NVIDIA CUDA SDK, development kit, and developer drivers be installed. These can be downloaded from: https://developer.nvidia.com/cuda-downloads

JCUDA interfaces with native CUDA libraries and must also be built from source for your system. On Ubuntu 12.04, the build process was as follows:

* Download JCUDA source from: http://www.jcuda.org/downloads/JCuda-All-0.5.0RC-src.zip
* export JAVA_HOME=/usr/lib/jvm/jdk1.6.0_32
* export PATH=$PATH:/usr/local/cuda-5.0/bin
* cmake -G "Unix Makefiles" CmakeLists.txt
* make

Running Project
============
To run the interactive heat map and histogram application, run: edu.gmu.ulman.histogram.HeatMapHistogramViewer.

To run the no-GUI test case (for profiling purposes), run: edu.gmu.ulman.histogram.HeatMapHistogramTest.

The application requires that CUDA and JCUDA are properly installed on the system. The required OpenGL native libraries are included in the Maven dependencies and will be added to java.library.path automatically. The JCUDA native libraries must be added manually. If running from Eclipse, add the following to the run configuration VM arguments:

* -Djava.library.path=JCuda-All-0.5.0RC-src/lib

