
# CUDA code generation flags
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   := $(GENCODE_SM20) $(GENCODE_SM30)

INCLUDES        := -I /usr/local/cuda/include -I /usr/local/cuda/samples/common/inc
LIBS            := -lglut -lGL -lcuda -lcudart -lm

histogram_test: histogram_test.o
	nvcc $(INCLUDES) $(GENCODE_FLAGS) $(LIBS) target/c/histogram_test.o -o target/c/histogram_test

histogram_test.o:
	nvcc $(INCLUDES) $(GENCODE_FLAGS) $(LIBS) -c src/main/c/histogram_test.cu -o target/c/histogram_test.o

clean:
	rm target/c/histogram_test.o target/c/histogram_test
