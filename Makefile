all: image.ppm

a.out: main.cu sphere.cpp
	@nvcc -arch=sm_86 main.cu sphere.cpp

image.ppm: a.out
	@./a.out image.ppm

clean:
	@rm -f a.out image.ppm
