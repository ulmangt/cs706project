
# CUDA code generation flags
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_FLAGS   := $(GENCODE_SM20) $(GENCODE_SM30)

INCLUDES        := -I /usr/local/cuda/include -I /usr/local/cuda/samples/common/inc
LIBS            := -lglut -lGL -lcuda -lcudart -lm

all: histogram_test histogram_test2 histogram_test3 histogram_test_nocuda

histogram_test_nocuda: histogram_test_nocuda.o
	gcc -lm target/c/histogram_test_nocuda.o -o target/c/histogram_test_nocuda

histogram_test_nocuda.o:
	gcc -lm -c src/main/c/histogram_test_nocuda.c -o target/c/histogram_test_nocuda.o

histogram_test: histogram_test.o
	nvcc $(INCLUDES) $(GENCODE_FLAGS) $(LIBS) target/c/histogram_test.o -o target/c/histogram_test

histogram_test.o:
	nvcc $(INCLUDES) $(GENCODE_FLAGS) $(LIBS) -c src/main/c/histogram_test.cu -o target/c/histogram_test.o

histogram_test2: histogram_test2.o
	nvcc $(INCLUDES) $(GENCODE_FLAGS) $(LIBS) target/c/histogram_test2.o -o target/c/histogram_test2

histogram_test2.o:
	nvcc $(INCLUDES) $(GENCODE_FLAGS) $(LIBS) -c src/main/c/histogram_test2.cu -o target/c/histogram_test2.o

histogram_test3: histogram_test3.o
	nvcc $(INCLUDES) $(GENCODE_FLAGS) $(LIBS) target/c/histogram_test3.o -o target/c/histogram_test3

histogram_test3.o:
	nvcc $(INCLUDES) $(GENCODE_FLAGS) $(LIBS) -c src/main/c/histogram_test3.cu -o target/c/histogram_test3.o

clean:
	rm -f target/c/histogram_test.o target/c/histogram_test target/c/histogram_test2.o target/c/histogram_test2 target/c/histogram_test3.o target/c/histogram_test3 target/c/histogram_test_nocuda.o target/c/histogram_test_nocuda
