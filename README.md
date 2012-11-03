cs706project
============

JCUDA histogram calculation with Glimpse data visualization.

Compilation
============

Maven can be used to automatically download dependency jars and compile this project.

CUDA also requires that the NVIDIA CUDA SDK, development kit, and developer drivers be installed. These can be downloaded from: https://developer.nvidia.com/cuda-downloads

JCUDA interfaces with native CUDA libraries and must also be built from source for your system. On Ubuntu 12.04, the build process was as follows:

Download JCUDA source from: http://www.jcuda.org/downloads/JCuda-All-0.5.0RC-src.zip
cmake -G "Unix Makefiles" CmakeLists.txt

