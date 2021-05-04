#include "hittable_list.hpp"

#include <cstdio>
#include <cuda.h>

void hittable_list::clear() {
    cudaFree(objects);
}

void hittable_list::add_spheres(size_t n, point3 *centers, float *radii) {
    clear();
    cudaMallocManaged(&objects, n * sizeof(hittable));
    _size = n;

    for (int i = 0; i < n; ++i) {
        objects[i] = hittable(sphere(centers[i], radii[i]));
    }
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
