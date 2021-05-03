#ifndef RAY_HPP
#define RAY_HPP

#include "vec3.hpp"

#ifndef __CUDACC__
    #define __device__
#endif

class ray {
    public:
        __device__ ray() {}
        __device__ ray(const point3& origin, const vec3& direction)
            : orig(origin), dir(direction)
        {}

        __device__ point3 origin() const { return orig; }
        __device__ vec3 direction() const { return dir; }

        __device__ point3 at(float t) const {
            return orig + (t * dir);
        }

    private:
        point3 orig;
        vec3 dir;
};

#ifndef __CUDACC__
    #undef __device__
#endif

#endif
