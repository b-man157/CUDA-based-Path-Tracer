OBJS = build/main.o build/hittable_list.o
CC = nvcc
LFLAGS = -std=c++17 -O3 -arch=sm_86
CFLAGS = $(LFLAGS) -dc -Xptxas -O5

all: image.ppm

image.ppm: bin/render.out
	bin/render.out image.ppm

bin/render.out: $(OBJS)
	$(CC) $(LFLAGS) $^ -o $@

build/main.o: src/main.cu
	$(CC) $(CFLAGS) $< -o $@ -I.

build/hittable_list.o: src/hittable_list/hittable_list.cu
	$(CC) $(CFLAGS) $< -o $@ -I.

clean:
	rm -f build/*.o bin/render.out image.ppm
