cs706project
============

JCUDA histogram calculation with Glimpse data visualization.

Java Compilation
============

Maven can be used to automatically download dependency jars and compile this project.

CUDA also requires that the NVIDIA CUDA SDK, development kit, and developer drivers be installed. These can be downloaded from: https://developer.nvidia.com/cuda-downloads

JCUDA interfaces with native CUDA libraries and must also be built from source for your system. On Ubuntu 12.04, the build process was as follows:

* Download JCUDA source from: http://www.jcuda.org/downloads/JCuda-All-0.5.0RC-src.zip
* export JAVA_HOME=/usr/lib/jvm/jdk1.6.0_32
* export PATH=$PATH:/usr/local/cuda-5.0/bin
* cmake -G "Unix Makefiles" CmakeLists.txt
* make

C Compilation
============

Because of problems encountered running the Nvidia Visual Profiler with JCUDA code, a simple headless C testbed application was created which calls the same kernel as the graphical java application. A makefile is included to compile this application. Running the application requires that /usr/local/cuda/lib64 be added to the LD_LIBRARY_PATH environemnt variable (on Ubuntu Linux).

Running Project
============
To run the interactive heat map and histogram application in Linux execute the runGUI.sh bash script.

To run the no-GUI test case (for profiling purposes) in Linux execute the runTest.sh bash script.

The application requires that CUDA and JCUDA are properly installed on the system. The required OpenGL native libraries are included in the Maven dependencies and will be added to java.library.path automatically. The JCUDA native libraries must be added manually. If running from Eclipse, add the following to the run configuration VM arguments:

* -Djava.library.path=JCuda-All-0.5.0RC-src/lib

Running from Eclipse also requires manually adding the JCUDA dependencies in JCuda-All-0.5.0RC-src/java-lib to the classpath, in addition to the Maven managed dependencies.

