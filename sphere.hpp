#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "hittable.hpp"
#include "vec3.hpp"

#ifdef __CUDACC__
    #define __HD__ __host__ __device__
#else
    #define __HD__
#endif

class sphere : public hittable {
    public:
        sphere() {}
        sphere(point3 cen, float r) : center(cen), radius(r) {}

        __HD__ bool hit(
            const ray &r, float t_min, float t_max, hit_record &rec) const override;

    private:
        point3 center;
        float radius;
};

#endif
