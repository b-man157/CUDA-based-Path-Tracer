#ifndef HITTABLE_LIST_HPP
#define HITTABLE_LIST_HPP

#include "include/hittable/hittable.hpp"

#ifndef __CUDACC__
    #define __device__
#endif

class hittable_list {
    public:
        hittable_list() : _size(0) {}

        void clear();
        void add_spheres(size_t n, point3 *centers, float *radii, material *materials);

        __device__ bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const;

    private:
        size_t _size;
        hittable *objects;
};

#ifndef __CUDACC__
    #undef __device__
#endif

#endif
