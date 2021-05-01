#ifndef HITTABLE_LIST_HPP
#define HITTABLE_LIST_HPP

#include "hittable.hpp"

#ifdef __CUDACC__
    #define __HD__ __host__ __device__
#else
    #define __HD__
#endif

class hittable_list : public hittable {
    public:
        __HD__ hittable_list() {
            #ifndef __CUDACC__
                _size = _capacity = 0;
                objects = NULL;
            #endif
        }

        hittable_list(hittable *object) : _size(0), _capacity(0), objects(NULL) {
            add(object);
        }

        __HD__ ~hittable_list() {
            clear();
        }

        __HD__ void clear() {
            delete[] objects;
        }

        void add(hittable *object);

        __HD__ virtual bool hit(
            const ray &r, float t_min, float t_max, hit_record &rec) const override;

    private:
        int _size, _capacity;
        hittable **objects;
};

#undef __HD__

#endif
