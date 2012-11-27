histogram_test: histogram_test.o
	nvcc -I /usr/local/cuda/include -I /usr/local/cuda/samples/common/inc -lglut -lGL -lcuda -lcudart -lm target/c/histogram_test.o -o target/c/histogram_test

histogram_test.o:
	nvcc -I /usr/local/cuda/include -I /usr/local/cuda/samples/common/inc -arch=sm_11 -lglut -lGL -lcuda -lcudart -lm -c src/main/c/histogram_test.cu -o target/c/histogram_test.o

clean:
	rm target/c/histogram_test.o target/c/histogram_test
