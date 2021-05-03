#ifndef HITTABLE_LIST_HPP
#define HITTABLE_LIST_HPP

#include "hittable.hpp"

class hittable_list {
    public:
        hittable_list() : _size(0) {}
        hittable_list(hittable *object);

        void clear();
        void add(hittable *object);

        bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const;

    private:
        size_t _size;
        hittable **objects;
};

#endif
