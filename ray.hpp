#ifndef RAY_HPP
#define RAY_HPP

#include "vec3.hpp"

#ifdef __CUDACC__
    #define __HD__ __host__ __device__
#else
    #define __HD__
#endif

class ray {
    public:
        __HD__ ray() {}
        __HD__ ray(const point3& origin, const vec3& direction)
            : orig(origin), dir(direction)
        {}

        __HD__ point3 origin() const { return orig; }
        __HD__ vec3 direction() const { return dir; }

        __HD__ point3 at(float t) const {
            return orig + (t * dir);
        }

    private:
        point3 orig;
        vec3 dir;
};

#undef __HD__

#endif
