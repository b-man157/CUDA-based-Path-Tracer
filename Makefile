OBJS = main.o hittable_list.o sphere.o
CC = nvcc
LFLAGS = -std=c++17 -O3 -arch=sm_86
CFLAGS = $(LFLAGS) -dc -Xptxas -O5

all: image.ppm

image.ppm: render.out
	./render.out image.ppm

render.out: $(OBJS)
	$(CC) $(LFLAGS) $^ -o $@

%.o: %.cu
	$(CC) $(CFLAGS) $< -o $@

%.o: %.cpp
	$(CC) -x cu $(CFLAGS) $< -o $@

clean:
	rm -f *.o render.out image.ppm
