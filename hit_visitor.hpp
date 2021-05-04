#ifndef HIT_VISITOR_HPP
#define HIT_VISITOR_HPP

#include "sphere.hpp"

#ifndef __CUDACC__
    #define __device__
#endif

class hit_visitor {
    public:
        __device__ hit_visitor(const ray &r, float t_min, float t_max, hit_record &rec) :
            r(r), t_min(t_min), t_max(t_max), rec(rec) {}

        using return_type = bool;

        __device__ return_type operator()(const sphere &s) {
            return s.hit(r, t_min, t_max, rec);
        }

    private:
        ray r;
        float t_min, t_max;
        hit_record &rec;
};

#ifndef __CUDACC__
    #undef __device__
#endif

#endif
