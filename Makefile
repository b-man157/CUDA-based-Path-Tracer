objects = main.o hittable_list.o sphere.o

all: image.ppm

image.ppm: a.out
	./a.out image.ppm

a.out: $(objects)
	nvcc -arch=sm_86 $(objects) -g -G

%.o: %.cu
	nvcc -arch=sm_86 -dc $< -o $@ -g -G

%.o: %.cpp
	nvcc -x cu -arch=sm_86 -dc $< -o $@ -g -G

clean:
	rm -f *.o a.out image.ppm
