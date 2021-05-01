#ifndef HITTABLE_HPP
#define HITTABLE_HPP

#include "ray.hpp"

#ifdef __CUDACC__
    #define __HD__ __host__ __device__
#else
    #define __HD__
#endif

struct hit_record {
    point3 p;
    vec3 normal;
    float t;
    bool front_face;

    __HD__ inline void set_face_normal(const ray &r, const vec3 &outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
    public:
        __HD__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const = 0;
};

#undef __HD__

#endif
