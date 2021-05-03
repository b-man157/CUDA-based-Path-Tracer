#ifndef HIT_VISITOR_HPP
#define HIT_VISITOR_HPP

#include "sphere.hpp"
#include "variant.hpp"

class hit_visitor : Visitor {
    public:
        hit_visitor(const ray &r, float t_min, float t_max, hit_record &rec) :
            r(r), t_min(t_min), t_max(t_max), rec(rec) {}

        auto operator()(sphere s) {
            return s.hit(r, t_min, t_max, rec);
        }

    private:
        ray r;
        float t_min, t_max;
        hit_record rec;
};

#endif
