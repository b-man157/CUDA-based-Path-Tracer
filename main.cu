#include "rtweekend.hpp"

#include "camera.hpp"
#include "color.hpp"
#include "hittable_list.hpp"

#include <cuda.h>
#include <curand_kernel.h>

#include <fstream>
#include <iostream>
#include <numeric>

__constant__ unsigned seed = 42;

__global__ void setup_curand(curandState *state, int image_width, int image_height) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < image_width && idy < image_height) {
        auto id = idy * image_width + idx;
        curand_init(seed, id, 0, &state[id]);
    }
}

__device__ color ray_color(const ray &r, const hittable_list *world) {
    hit_record rec;
    if (world->hit(r, 0, infinity, rec)) {
        return 0.5 * (rec.normal + color(1, 1, 1));
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

inline __device__ float random_float(curandState *local_state) {
    // Returns a random real in (0, 1].
    return curand_uniform(local_state);
}

__global__ void render(
        curandState *state, const camera *cam,
        int image_width, int image_height,
        const hittable_list *world,
        color *pixel_colors, int samples_per_pixel) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < image_width && idy < image_height) {
        auto id = idy * image_width + idx;
        auto p_index = (image_height-1 - idy) * image_width + idx;
        pixel_colors[p_index] = color(0, 0, 0);

        for (int i = 0; i < samples_per_pixel; ++i) {
            float u = (idx + random_float(&state[id])) / (image_width-1);
            float v = (idy + random_float(&state[id])) / (image_height-1);
            auto r = cam->get_ray(u, v);

            pixel_colors[p_index] += ray_color(r, world);
        }
    }
}

int main(int argc, char **argv) {
    // Arguments

    if (argc != 2) {
        std::cerr << "Specify output file.\n";
        return -1;
    }

    // Image

    const float aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 100;

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

    const size_t n_spheres = 2;
    point3 centers[] = {{0, 0, -1}, {0, -100.5, -1}};
    float radii[] = {0.5, 100};

    hittable_list h_world;
    h_world.add_spheres(n_spheres, centers, radii);

    // Camera

    camera h_cam, *d_cam;
    cudaMalloc(&d_cam, sizeof(camera));
    cudaMemcpyAsync(d_cam, &h_cam, sizeof(camera), cudaMemcpyHostToDevice);

    // Render

    std::ofstream f_out(argv[1]);

    f_out << "P3\n" << image_width << ' ' << image_height << "\n255";

    const int n_pixels = image_width * image_height;
    color *d_pixels;
    cudaMalloc(&d_pixels, n_pixels * sizeof(color));

    hittable_list *d_world;
    cudaMalloc(&d_world, sizeof(hittable_list));
    cudaMemcpyAsync(d_world, &h_world, sizeof(hittable_list), cudaMemcpyHostToDevice);

    curandState *d_state;
    cudaMalloc(&d_state, n_pixels * sizeof(curandState));
    setup_curand<<<grid_dim, block_dim>>>(d_state, image_width, image_height);

    // TODO: Track and print progress.
    render<<<grid_dim, block_dim>>>(
        d_state, d_cam, image_width, image_height, d_world, d_pixels, samples_per_pixel);

    h_world.clear();
    cudaFree(d_state);
    cudaFree(d_cam);
    cudaFree(d_world);

    color *h_pixels = (color *) std::malloc(n_pixels * sizeof(color));
    cudaMemcpy(h_pixels, d_pixels, n_pixels * sizeof(color), cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);

    f_out << std::accumulate(h_pixels, h_pixels + n_pixels, std::string(""),
        [samples_per_pixel](const std::string s, const color c) {
            return s + '\n' + to_string(c, samples_per_pixel);
        }
    );
    std::free(h_pixels);

    f_out.close();

    return 0;
}
