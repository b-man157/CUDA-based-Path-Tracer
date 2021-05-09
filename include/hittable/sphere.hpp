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

        __device__ bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const {
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
            rec.mat_ptr = mat_ptr;

            return true;
        }

    private:
        point3 center;
        float radius;
        material *mat_ptr;
};

#ifndef __CUDACC__
    #undef __device__
#endif

#endif
