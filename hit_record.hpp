#ifndef HIT_RECORD_HPP
#define HIT_RECORD_HPP

#include "ray.hpp"

#ifndef __CUDACC__
    #define __device__
#endif

// Forward-declaration of material.

template <typename ...Ts>
class generic_material;

class dielectric;
class lambertian;
class metal;

typedef generic_material<dielectric, lambertian, metal> material;

// End.

struct hit_record {
    point3 p;
    vec3 normal;
    material *mat_ptr;
    double t;
    bool front_face;

    __device__ inline void set_face_normal(const ray &r, const vec3 &outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

#ifndef __CUDACC__
    #undef __device__
#endif

#endif
