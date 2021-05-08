#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "hit_record.hpp"

#ifndef __CUDACC__
    #define __device__
#endif

class sphere {
    public:
        sphere() {}
        __device__ sphere(point3 cen, float r, material *m)
            : center(cen), radius(r), mat_ptr(m) {}

        __device__ bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const;

    private:
        point3 center;
        float radius;
        material *mat_ptr;
};

#ifndef __CUDACC__
    #undef __device__
#endif

#endif
