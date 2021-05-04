#ifndef HITTABLE_HPP
#define HITTABLE_HPP

#include "hit_visitor.hpp"
#include "variant.hpp"

#ifndef __CUDACC__
    #define __device__
#endif

template <typename ...Ts>
class generic_hittable : public Variant<Ts...> {
    public:
        generic_hittable() = default;

        template <typename Obj>
        generic_hittable(Obj obj) : Variant<Ts...>(obj) {}

        __device__ hit_visitor::return_type hit(
                const ray &r, float t_min, float t_max, hit_record &rec) const {
            auto hv = hit_visitor(r, t_min, t_max, rec);
            return applyVisitor(hv, *this);
        }
};

typedef generic_hittable<sphere> hittable;

#ifndef __CUDACC__
    #undef __device__
#endif

#endif
