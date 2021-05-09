#ifndef LAMBERTIAN_HPP
#define LAMBERTIAN_HPP

#include "include/hittable/hit_record.hpp"

#ifndef __CUDACC__
    #define __device__
#endif

class lambertian {
    public:
        lambertian(const color &a) : albedo(a) {}

        template <typename curandState>
        __device__ bool scatter(
            curandState *state, const ray &r_in, const hit_record &rec,
            color &attenuation, ray &scattered
        ) const {
            auto scatter_direction = rec.normal + random_unit_vector(state);

            // Catch degenerate scatter direction.
            if (scatter_direction.near_zero())
                scatter_direction = rec.normal;

            scattered = ray(rec.p, scatter_direction);
            attenuation = albedo;
            return true;
        }

    private:
        color albedo;
};

#ifndef __CUDACC__
    #undef __device__
#endif

#endif
