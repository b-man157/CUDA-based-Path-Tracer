#include "rtweekend.hpp"

#include "include/vec3/camera.hpp"
#include "include/vec3/color.hpp"
#include "include/material/material.hpp"
#include "src/hittable_list/hittable_list.hpp"

#include <cuda.h>

#include <fstream>
#include <numeric>
#include <vector>

const unsigned h_seed = 42;
__constant__ unsigned d_seed = 42;

const bool USE_STREAMS = false;
const unsigned NTHREADS = 128, NSTREAMS = 4;

__global__ void setup_curand(curandState *state, int image_width, int image_height) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < image_width && idy < image_height) {
        auto id = idy * image_width + idx;
        curand_init(d_seed, id, 0, &state[id]);
    }
}

__global__ void clear_pixels(const int image_width, const int image_height, color *pixel_colors) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < image_width && idy < image_height) {
        auto p_index = (image_height-1 - idy) * image_width + idx;
        pixel_colors[p_index] = color(0, 0, 0);
    }
}

template <int depth>
__device__ color ray_color(
        curandState *local_state, const ray &r, const hittable_list *world) {
    hit_record rec;
    // Fix shadow acne by setting t_min > 0.
    if (world->hit(r, 0.001, infinity, rec)) {
        ray scattered;
        color attenuation;
        if (rec.mat_ptr->scatter(local_state, r, rec, attenuation, scattered))
            return attenuation * ray_color<depth - 1>(local_state, scattered, world);
        return color(0, 0, 0);
    }

    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

template <>
__device__ color ray_color<0>(
        curandState *local_state, const ray &r, const hittable_list *world) {
    return color(0, 0, 0);
}

template <int max_depth>
__global__ void render(
        curandState *state, const camera *cam, const int image_width, const int image_height,
        const hittable_list *world, color *pixel_colors, int samples_per_pixel) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < image_width && idy < image_height) {
        auto id = idy * image_width + idx;
        auto p_index = (image_height-1 - idy) * image_width + idx;
        auto pixel_color = color(0, 0, 0);

        for (int i = 0; i < samples_per_pixel; ++i) {
            float u = (idx + random_float(&state[id])) / (image_width-1);
            float v = (idy + random_float(&state[id])) / (image_height-1);
            auto r = cam->get_ray(&state[id], u, v);

            pixel_colors[p_index] += ray_color<max_depth>(&state[id], r, world);
        }
        atomicAdd(&pixel_colors[p_index][0], pixel_color[0]);
        atomicAdd(&pixel_colors[p_index][1], pixel_color[1]);
        atomicAdd(&pixel_colors[p_index][2], pixel_color[2]);
    }
}

hittable_list random_scene() {
    hittable_list world;

    std::vector<point3> centers;
    std::vector<float> radii;
    std::vector<material> materials;
    centers.reserve(488);
    radii.reserve(488);
    materials.reserve(488);

    material ground_material = lambertian(color(0.5, 0.5, 0.5));
    centers.push_back(point3(0, -1000, 0));
    radii.push_back(1000);
    materials.push_back(ground_material);

    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            float choose_mat = random_float();
            point3 center(a + 0.9*random_float(), 0.2, b + 0.9*random_float());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                material sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = lambertian(albedo);
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    float fuzz = random_float(0, 0.5);
                    sphere_material = metal(albedo, fuzz);
                } else {
                    // glass
                    sphere_material = dielectric(1.5);
                }

                centers.push_back(center);
                radii.push_back(0.2);
                materials.push_back(sphere_material);
            }
        }
    }

    material material1 = dielectric(1.5);
    centers.push_back(point3(0, 1, 0));
    radii.push_back(1.0);
    materials.push_back(material1);

    material material2 = lambertian(color(0.4, 0.2, 0.1));
    centers.push_back(point3(-4, 1, 0));
    radii.push_back(1.0);
    materials.push_back(material2);

    material material3 = metal(color(0.7, 0.6, 0.5), 0.0);
    centers.push_back(point3(4, 1, 0));
    radii.push_back(1.0);
    materials.push_back(material3);

    centers.shrink_to_fit();
    radii.shrink_to_fit();
    materials.shrink_to_fit();
    world.add_spheres(centers.size(), centers.data(), radii.data(), materials.data());

    return world;
}

int main(int argc, char **argv) {
    // Arguments

    if (argc != 2) {
        std::cerr << "Specify output file.\n";
        return -1;
    }

    // Image

    const float aspect_ratio = 3.0 / 2.0;
    const int image_width = 1200;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 500;
    const int max_depth = 50;

    // Dimensions

    int x = sqrt(NTHREADS * aspect_ratio); x = (x > 32) ? (x - x % 32) : 32;
    int y = NTHREADS / x;

    dim3 block_dim;
    block_dim.x = x;
    block_dim.y = y;
    block_dim.z = 1;

    dim3 grid_dim;
    grid_dim.x = (image_width / x) + (image_width % x > 0);
    grid_dim.y = (image_height / y) + (image_height % y > 0);
    grid_dim.z = 1;

    // World

    srand(h_seed);
    hittable_list h_world = random_scene();

    hittable_list *d_world;
    cudaMalloc(&d_world, sizeof(hittable_list));
    cudaMemcpyAsync(d_world, &h_world, sizeof(hittable_list), cudaMemcpyHostToDevice);

    // Camera

    point3 lookfrom(13, 2, 3);
    point3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    float dist_to_focus = 10;
    float aperture = 0.1;
    camera h_cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

    camera *d_cam;
    cudaMalloc(&d_cam, sizeof(camera));
    cudaMemcpyAsync(d_cam, &h_cam, sizeof(camera), cudaMemcpyHostToDevice);

    // Render

    std::ofstream f_out(argv[1]);

    f_out << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    const int n_pixels = image_width * image_height;
    color *d_pixels;
    cudaMalloc(&d_pixels, n_pixels * sizeof(color));

    curandState *d_state;
    cudaMalloc(&d_state, n_pixels * sizeof(curandState));
    setup_curand<<<grid_dim, block_dim>>>(d_state, image_width, image_height);

    if (USE_STREAMS) {
        cudaStream_t streams[NSTREAMS];
        for (int i = 0; i < NSTREAMS; ++i) {
            cudaStreamCreate(&streams[i]);
        }
        clear_pixels<<<grid_dim, block_dim>>>(image_width, image_height, d_pixels);

        for (int i = 0; i < NSTREAMS; ++i) {
            render<max_depth><<<grid_dim, block_dim, 0, streams[i]>>>(
                d_state, d_cam, image_width, image_height,
                d_world, d_pixels, samples_per_pixel/NSTREAMS + 1
            );
        }

        for (int i = 0; i < NSTREAMS; ++i) {
            cudaStreamDestroy(streams[i]);
        }
        cudaDeviceSynchronize();
    }
    else {
        clear_pixels<<<grid_dim, block_dim>>>(image_width, image_height, d_pixels);
        render<max_depth><<<grid_dim, block_dim>>>(
            d_state, d_cam, image_width, image_height,
            d_world, d_pixels, samples_per_pixel
        );
    }

    h_world.clear();
    cudaFree(d_state);
    cudaFree(d_cam);
    cudaFree(d_world);

    color *h_pixels = (color *) std::malloc(n_pixels * sizeof(color));
    cudaMemcpy(h_pixels, d_pixels, n_pixels * sizeof(color), cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);

    for (int i = 0; i < n_pixels-1; ++i) {
        write_color(f_out, h_pixels[i], samples_per_pixel);
    }
    f_out << to_string(h_pixels[n_pixels-1], samples_per_pixel);
    std::free(h_pixels);

    f_out.close();

    return 0;
}
