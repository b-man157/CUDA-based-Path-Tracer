#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "ray.hpp"

#ifndef __CUDACC__
    #define __device__
#endif

class camera {
    public:
        camera(int width, float ratio) :
                image_width(width), image_height(width / ratio), aspect_ratio(ratio) {
            float viewport_width = 2.0 * ratio, viewport_height = 2.0;
            float focal_length = 1.0;

            origin = point3(0, 0, 0);
            horizontal = vec3(viewport_width, 0, 0);
            vertical = vec3(0, viewport_height, 0);
            lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);
        }

        __device__ ray get_ray(int x, int y) const {
            float u = float(x) / (image_width - 1);
            float v = float(y) / (image_height - 1);
            return ray(origin, lower_left_corner + u*horizontal + v*vertical - origin);
        }

    public:
        const int image_width, image_height;
        const float aspect_ratio;

    private:
        point3 origin, lower_left_corner;
        vec3 horizontal, vertical;
};

#endif
