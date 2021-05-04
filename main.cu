#include "rtweekend.hpp"

#include "camera.hpp"
#include "color.hpp"
#include "hittable_list.hpp"
#include "sphere.hpp"

#include <cuda.h>

#include <fstream>
#include <iostream>
#include <numeric>

__device__ color ray_color(const ray &r, const hittable_list *world) {
    hit_record rec;
    if (world->hit(r, 0, infinity, rec)) {
        return 0.5 * (rec.normal + color(1, 1, 1));
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

__global__ void render(const camera *setup, const hittable_list *world, color *pixel_colors) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;

    auto i_width = setup->image_width, i_height = setup->image_height;

    if (idx < i_width && idy < i_height) {
        auto r = setup->get_ray(idx, idy);
        auto p_index = (i_height-1 - idy) * i_width + idx;
        pixel_colors[p_index] = ray_color(r, world);
    }
}

int main(int argc, char **argv) {
    // Arguments

    if (argc != 2) {
        std::cerr << "Specify output file.\n";
        return -1;
    }

    // Image

    camera h_setup(400, 16.0 / 9.0);

    // Dimensions

    dim3 block_dim;
    block_dim.x = sqrt(1024 * h_setup.aspect_ratio);
    block_dim.y = block_dim.x / h_setup.aspect_ratio;
    block_dim.z = 1;

    dim3 grid_dim;
    grid_dim.x = (h_setup.image_width  / block_dim.x) + (h_setup.image_width  % block_dim.x > 0);
    grid_dim.y = (h_setup.image_height / block_dim.y) + (h_setup.image_height % block_dim.y > 0);
    grid_dim.z = 1;

    // World

    const size_t n_spheres = 2;
    point3 centers[] = {{0, 0, -1}, {0, -100.5, -1}};
    float radii[] = {0.5, 100};

    hittable_list h_world;
    h_world.add_spheres(n_spheres, centers, radii);

    // Camera

    // TODO: Nothing here for now.

    // Render

    std::ofstream f_out(argv[1]);

    f_out << "P3\n" << h_setup.image_width << ' ' << h_setup.image_height << "\n255";

    const int n_pixels = h_setup.image_width * h_setup.image_height;
    color *d_pixels;
    cudaMalloc(&d_pixels, n_pixels * sizeof(color));

    camera *d_setup;
    cudaMalloc(&d_setup, sizeof(camera));
    cudaMemcpy(d_setup, &h_setup, sizeof(camera), cudaMemcpyHostToDevice);

    hittable_list *d_world;
    cudaMalloc(&d_world, sizeof(hittable_list));
    cudaMemcpy(d_world, &h_world, sizeof(hittable_list), cudaMemcpyHostToDevice);

    // TODO: Track and print progress.
    render<<<grid_dim, block_dim>>>(d_setup, d_world, d_pixels);

    color *h_pixels = (color *) std::malloc(n_pixels * sizeof(color));
    cudaMemcpy(h_pixels, d_pixels, n_pixels * sizeof(color), cudaMemcpyDeviceToHost);
    h_world.clear();

    f_out << std::accumulate(h_pixels, h_pixels + n_pixels, std::string(""),
        [](const std::string s, const color c) {
            return s + '\n' + to_string(c);
        }
    );

    f_out.close();

    return 0;
}
