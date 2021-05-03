#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "hit_record.hpp"

class sphere {
    public:
        sphere() {}
        sphere(point3 cen, float r) : center(cen), radius(r) {}

        bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const;

    private:
        point3 center;
        float radius;
};

#endif
