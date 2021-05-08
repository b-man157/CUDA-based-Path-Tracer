#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "ray.hpp"

#ifndef __CUDACC__
    #define __device__
#endif

class camera {
    public:
        camera(
            point3 lookfrom,
            point3 lookat,
            vec3 vup,
            float vfov,         // Vertical field-of-view, in degrees.
            float aspect_ratio
        ) {
            float theta = degrees_to_radians(vfov);
            float h = tan(theta/2);
            float viewport_height = 2.0 * h;
            float viewport_width = aspect_ratio * viewport_height;

            auto w = unit_vector(lookfrom - lookat);
            auto u = unit_vector(cross(vup, w));
            auto v = cross(w, u);

            origin = lookfrom;
            horizontal = viewport_width * u;
            vertical = viewport_height * v;
            lower_left_corner = origin - horizontal/2 - vertical/2 - w;
        }

        __device__ ray get_ray(float s, float t) const {
            return ray(origin, lower_left_corner + s*horizontal + t*vertical - origin);
        }

    private:
        point3 origin;
        point3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
};

#endif
