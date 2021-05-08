#ifndef SCATTER_VISITOR_HPP
#define SCATTER_VISITOR_HPP

#include "dielectric.hpp"
#include "lambertian.hpp"
#include "metal.hpp"

#include <curand_kernel.h>

#ifndef __CUDACC__
    #define __device__
    class curandState;
#endif

class scatter_visitor {
    public:
        __device__ scatter_visitor(
            curandState *state, const ray &r_in, const hit_record &rec,
            color &attenuation, ray &scattered)
            : state(state), r_in(r_in), rec(rec),
            attenuation(attenuation), scattered(scattered) {}

        using return_type = bool;

        __device__ return_type operator()(const dielectric &d) {
            return d.scatter(state, r_in, rec, attenuation, scattered);
        }

        __device__ return_type operator()(const lambertian &l) {
            return l.scatter(state, r_in, rec, attenuation, scattered);
        }

        __device__ return_type operator()(const metal &m) {
            return m.scatter(state, r_in, rec, attenuation, scattered);
        }

    private:
        curandState *state;
        const ray &r_in;
        const hit_record &rec;
        color &attenuation;
        ray &scattered;
};

#ifndef __CUDACC__
    #undef __device__
#endif

#endif
