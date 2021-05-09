#ifndef COLOR_HPP
#define COLOR_HPP

#include "vec3.hpp"

#include <iostream>
#include <string>

inline std::string to_string(color pixel_color, int samples_per_pixel) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Divide the color by the number of samples and gamma-correct for gamma=2.0.
    auto scale = 1.0 / samples_per_pixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    // Return the translated [0, 255] value of each color component.
    return std::to_string(static_cast<int>(256 * clamp(r, 0.0, 0.999))) + ' '
         + std::to_string(static_cast<int>(256 * clamp(g, 0.0, 0.999))) + ' '
         + std::to_string(static_cast<int>(256 * clamp(b, 0.0, 0.999)));
}

inline void write_color(
        std::ostream &out, const color &pixel_color, int samples_per_pixel) {
    out << to_string(pixel_color, samples_per_pixel) << '\n';
}

#endif
