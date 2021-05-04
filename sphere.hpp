#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "hit_record.hpp"

#ifndef __CUDACC__
    #define __device__
#endif

class sphere {
    public:
        sphere() {}
        __device__ sphere(point3 cen, float r) : center(cen), radius(r) {}

        __device__ bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const;

    private:
        point3 center;
        float radius;
};

#ifndef __CUDACC__
    #undef __device__
#endif

#endif
