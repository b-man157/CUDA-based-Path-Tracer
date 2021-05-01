#ifndef COLOR_HPP
#define COLOR_HPP

#include "vec3.hpp"

#include <iostream>
#include <string>

inline std::string to_string(const color &pixel_color) {
    // First translates to the [0, 255] value of each color component.
    return std::to_string(static_cast<int>(255.999 * pixel_color.x())) + ' '
         + std::to_string(static_cast<int>(255.999 * pixel_color.y())) + ' '
         + std::to_string(static_cast<int>(255.999 * pixel_color.z()));
}

inline void write_color(std::ostream &out, const color &pixel_color) {
    out << to_string(pixel_color) << '\n';
}

#endif
