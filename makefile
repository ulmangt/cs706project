histogram_test: histogram_test.o HistogramTextureKernel.o
	nvcc -lcuda -lcudart -lm target/c/histogram_test.o target/c/HistogramTextureKernel.o -o target/c/histogram_test

histogram_test.o:
	gcc -c src/main/c/histogram_test.c -o target/c/histogram_test.o

HistogramTextureKernel.o:
	nvcc -arch=sm_11 -lcuda -lcudart -lm -c src/main/java/resources/HistogramTextureKernel.cu -o target/c/HistogramTextureKernel.o

clean:
	rm target/c/histogram_test.o target/c/HistogramTextureKernel.o target/c/histogram_test
