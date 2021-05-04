#include "hittable_list.hpp"

#include <cuda.h>

void hittable_list::clear() {
    if (_size) {
        cudaFree(objects);
        _size = 0;
    }
}

__global__ void add_spheres(size_t n, point3 *centers, float *radii, hittable *spheres) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        spheres[idx] = hittable(sphere(centers[idx], radii[idx]));
    }
}

void hittable_list::add_spheres(size_t n, point3 *centers, float *radii) {
    clear();
    cudaMalloc(&objects, n * sizeof(hittable));
    _size = n;

    point3 *d_centers;
    float *d_radii;
    cudaMalloc(&d_centers, n * sizeof(point3));
    cudaMalloc(&d_radii, n * sizeof(float));

    cudaMemcpyAsync(d_centers, centers, n * sizeof(point3), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_radii, radii, n * sizeof(float), cudaMemcpyHostToDevice);

    size_t n_blks = (n / 1024) + (n % 1024 > 0);
    ::add_spheres<<<n_blks, 1024>>>(n, d_centers, d_radii, objects);

    cudaFree(d_centers);
    cudaFree(d_radii);
}

__device__ bool hittable_list::hit(
        const ray &r, float t_min, float t_max, hit_record &rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (int i = 0; i < _size; ++i) {
        if (objects[i].hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}
