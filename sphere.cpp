#include "sphere.hpp"
#include "hittable.hpp"

#ifndef __CUDACC__
    #define __device__
#endif

__device__ bool sphere::hit(const ray &r, float t_min, float t_max, hit_record &rec) const {
    auto oc = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius*radius;

    float discriminant = half_b*half_b - a*c;
    if (discriminant < 0.0) return false;
    float sqrtd = sqrt(discriminant);

    // Find the nearest root that lies int the acceptable range.
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root) {
            return false;
        }
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    auto outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);

    return true;
}

#ifndef __CUDACC__
    #undef __device__
#endif
