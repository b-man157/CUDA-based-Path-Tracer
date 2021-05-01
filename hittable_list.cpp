#include "hittable_list.hpp"

void hittable_list::add(hittable *object) {
    if (_size == _capacity) {
        _capacity = std::max(1, 2 * _capacity);
        auto new_objects = new hittable *[_capacity];

        for (int i = 0; i < _size; ++i) {
            new_objects[i] = objects[i];
        }
        new_objects[_size] = object;
        ++_size;

        if (objects) delete[] objects;
        objects = new_objects;
    }
    else {
        objects[_size] = object;
        ++_size;
    }
}

bool hittable_list::hit(const ray &r, float t_min, float t_max, hit_record &rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < _size; ++i) {
        if (objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}
