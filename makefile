histogram_test: histogram_test.o HistogramTextureKernel.o
	nvcc -I /usr/local/cuda/include -lglut -lGL -lcuda -lcudart -lm target/c/histogram_test.o target/c/HistogramTextureKernel.o -o target/c/histogram_test

histogram_test.o:
	nvcc -I /usr/local/cuda/include -lglut -lGL -lcuda -lcudart -lm -c src/main/c/histogram_test.cu -o target/c/histogram_test.o

HistogramTextureKernel.o:
	nvcc -I /usr/local/cuda/include -arch=sm_11 -lglut -lGL -lcuda -lcudart -lm -c src/main/java/resources/HistogramTextureKernel.cu -o target/c/HistogramTextureKernel.o

clean:
	rm target/c/histogram_test.o target/c/HistogramTextureKernel.o target/c/histogram_test
