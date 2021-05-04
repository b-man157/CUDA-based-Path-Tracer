objects = main.o hittable_list.o sphere.o

all: image.ppm

image.ppm: a.out
	./a.out image.ppm

a.out: $(objects)
	nvcc -arch=sm_86 $(objects)

%.o: %.cu
	nvcc -arch=sm_86 -dc $< -o $@

%.o: %.cpp
	nvcc -x cu -arch=sm_86 -dc $< -o $@

clean:
	rm -f *.o a.out image.ppm
