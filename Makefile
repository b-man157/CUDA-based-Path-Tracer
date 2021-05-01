all: image.ppm

a.out: main.cu
	@nvcc main.cu

image.ppm: a.out
	@./a.out image.ppm

clean:
	@rm -f a.out image.ppm
