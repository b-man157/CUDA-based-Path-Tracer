#ifndef DIELECTRIC_HPP
#define DIELECTRIC_HPP

#include "include/hittable/hit_record.hpp"

#ifndef __CUDACC__
    #define __device__
#endif

class dielectric {
    public:
        dielectric(float index_of_refraction) : ir(index_of_refraction) {}

        __device__ static float reflectance(float cosine, float ref_idx) {
            // Use Schlick's approximation for reflectance.
            float r0 = (1 - ref_idx) / (1 + ref_idx);
            r0 = r0*r0;
            return r0 + (1 - r0) * pow((1 - cosine), 5);
        }

        template <typename curandState>
        __device__ bool scatter(
            curandState *state, const ray &r_in, const hit_record &rec,
            color &attenuation, ray &scattered
        ) const {
            attenuation = color(1.0, 1.0, 1.0);
            float refraction_ratio = rec.front_face ? (1.0/ir) : ir;

            vec3 unit_direction = unit_vector(r_in.direction());
            float cos_theta = fmin(dot(-unit_direction, rec.normal), (float) 1.0);
            float sin_theta = sqrt(1.0 - cos_theta*cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > 1.0;
            vec3 direction;
            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float(state))
                direction = reflect(unit_direction, rec.normal);
            else
                direction = refract(unit_direction, rec.normal, refraction_ratio);

            scattered = ray(rec.p, direction);
            return true;
        }

    private:
        float ir;       // Index of Refraction.
};

#ifndef __CUDACC__
    #undef __device__
#endif

#endif
