#include "hittable.hpp"
#include "rtweekend.hpp"

#include "color.hpp"
#include "hittable_list.hpp"
#include "sphere.hpp"

#include <cuda.h>

#include <fstream>
#include <iostream>
#include <numeric>

__device__ float hit_sphere(const point3 &center, float radius, const ray &r) {
    auto oc = r.origin() - center;
    float a = r.direction().length_squared();
    float half_b = dot(oc, r.direction());
    float c = oc.length_squared() - radius*radius;
    float discriminant = half_b * half_b - a * c;

    if (discriminant < 0.0) {
        return -1.0;
    }
    else {
        return (-half_b - sqrt(discriminant)) / a;
    }
}

__device__ color ray_color(const ray &r, const hittable &world) {
    hit_record rec;
    if (world.hit(r, 0, infinity, rec)) {
        return 0.5 * (rec.normal + color(1, 1, 1));
    }
    //
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

// TODO: Better name?
__global__ void render(int i_width, int i_height, float v_width, float v_height, float f_length, color *pixel_colors) {
    __shared__ vec3 origin, horizontal, vertical, lower_left_corner;

    if (!threadIdx.x && !threadIdx.y) {
        origin = point3(0, 0, 0);
        horizontal = vec3(v_width, 0, 0);
        vertical = vec3(0, v_height, 0);
        lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, f_length);
    }
    __syncthreads();

    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < i_width && idy < i_height) {
        float u = float(idx) / (i_width-1);
        float v = float(idy) / (i_height-1);
        ray r(origin, lower_left_corner + u*horizontal + v*vertical - origin);

        auto p_index = (i_height-1 - idy) * i_width + idx;
        pixel_colors[p_index] = ray_color(r, world);
    }
}

sphere *make_sphere(point3 center, int radius) {
    sphere *ptr = new sphere;
    return ptr;
}

int main(int argc, char **argv) {
    // Arguments

    if (argc != 2) {
        std::cerr << "Specify output file.\n";
        return -1;
    }

    // Image

    const float aspect_ratio = 16.0 / 9.0;
    const int image_width  = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    // Dimensions

    dim3 block_dim;
    block_dim.x = sqrt(1024 * aspect_ratio);
    block_dim.y = block_dim.x / aspect_ratio;
    block_dim.z = 1;

    dim3 grid_dim;
    grid_dim.x = (image_width  / block_dim.x) + (image_width  % block_dim.x > 0);
    grid_dim.y = (image_height / block_dim.y) + (image_height % block_dim.y > 0);
    grid_dim.z = 1;

    // World

    hittable_list world;
    auto s = sphere(point3(0, 0, 0), 5);
    world.add(make_sphere(point3(0,    0.0, -1),   0.5));
    world.add(make_sphere(point3(0, -100.5, -1), 100.0));

    // Camera

    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0;

    // Render

    std::ofstream f_out(argv[1]);

    f_out << "P3\n" << image_width << ' ' << image_height << "\n255";

    const int n_pixels = image_width * image_height;
    color *d_pixel_colors, *h_pixel_colors = (color *) std::malloc(n_pixels * sizeof(color));
    cudaMalloc(&d_pixel_colors, n_pixels * sizeof(color));

    // TODO: Track and print progress.
    render<<<grid_dim, block_dim>>>(image_width, image_height, viewport_width, viewport_height, focal_length, d_pixel_colors);
    cudaMemcpy(h_pixel_colors, d_pixel_colors, n_pixels * sizeof(color), cudaMemcpyDeviceToHost);

    f_out << std::accumulate(h_pixel_colors, h_pixel_colors + n_pixels, std::string(""),
        [](const std::string s, const color c) {
            return s + '\n' + to_string(c);
        }
    );

    f_out.close();

    return 0;
}
