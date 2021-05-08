#ifndef METAL_HPP
#define METAL_HPP

#include "hit_record.hpp"

#ifndef __CUDACC__
    #define __device__
#endif

class metal {
    public:
        metal(const color &a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

        template <typename curandState>
        __device__ bool scatter(
            curandState *state, const ray &r_in, const hit_record &rec,
            color &attenuation, ray &scattered
        ) const {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(state));
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }

    private:
        color albedo;
        float fuzz;
};

#ifndef __CUDACC__
    #undef __device__
#endif

#endif
