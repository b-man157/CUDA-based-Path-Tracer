#ifndef HITTABLE_HPP
#define HITTABLE_HPP

#include "hit_visitor.hpp"

template <typename ...Ts>
class generic_hittable : public Variant<Ts...> {
    public:
        template <typename Obj>
        generic_hittable(Obj obj) : Variant<Ts...>(obj) {}

        bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const {
            return applyVisitor<Ts...>(hit_visitor(r, t_min, t_max, rec), *this);
        }
};

typedef generic_hittable<sphere> hittable;

#endif
