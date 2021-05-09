#include "hittable_list.hpp"
#include "material.hpp"

#include <cuda.h>

const int NTHREADS = 32;

void hittable_list::clear() {
    if (_size) {
        cudaFree(objects);
        _size = 0;
    }
}

__global__ void add_spheres(
        size_t n, point3 *centers, float *radii, material *materials,
        hittable *spheres) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        spheres[idx] = hittable(sphere(centers[idx], radii[idx], &materials[idx]));
    }
}

void hittable_list::add_spheres(size_t n, point3 *centers, float *radii, material *materials) {
    clear();
    cudaMalloc(&objects, n * sizeof(hittable));
    _size = n;

    point3 *d_centers;
    float *d_radii;
    material *d_materials;
    cudaMalloc(&d_centers, n * sizeof(point3));
    cudaMalloc(&d_radii, n * sizeof(float));
    cudaMalloc(&d_materials, n * sizeof(material));

    cudaMemcpyAsync(d_centers, centers, n * sizeof(point3), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_radii, radii, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_materials, materials, n * sizeof(material), cudaMemcpyHostToDevice);

    size_t n_blks = (n / NTHREADS) + (n % NTHREADS > 0);
    ::add_spheres<<<n_blks, NTHREADS>>>(n, d_centers, d_radii, d_materials, objects);

    cudaFree(d_centers);
    cudaFree(d_radii);
    // d_materials not freed as it could be reused.
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
