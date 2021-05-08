#ifndef MATERIAL_HPP
#define MATERIAL_HPP

#include "rtweekend.hpp"

#include "hit_record.hpp"
#include "scatter_visitor.hpp"
#include "variant.hpp"

#ifndef __CUDACC__
    #define __device__
#endif

template <typename ...Ts>
class generic_material : public Variant<Ts...> {
    public:
        generic_material() = default;

        template <typename Mat>
        generic_material(Mat mat) : Variant<Ts...>(mat) {}

        template <typename curandState>
        __device__ scatter_visitor::return_type scatter(
            curandState *state, const ray &r_in, const hit_record &rec,
            color &attenuation, ray &scattered
        ) const {
            auto sv = scatter_visitor(state, r_in, rec, attenuation, scattered);
            return applyVisitor(sv, *this);
        }
};

typedef generic_material<dielectric, lambertian, metal> material;

#ifndef __CUDACC__
    #undef __device__
#endif

#endif
