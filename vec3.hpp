#ifndef VEC3_HPP
#define VEC3_HPP

#include "rtweekend.hpp"

#include <iostream>

#ifdef __CUDACC__
    #define __HD__ __host__ __device__
#else
    #define __host__
    #define __device__
    #define __HD__
#endif

using std::sqrt;

class vec3 {
    public:
        __HD__ vec3() {
            #ifndef __CUDA_ARCH__
                e[0] = e[1] = e[2] = 0;
            #endif
        }
        __HD__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

        __HD__ float x() const { return e[0]; }
        __HD__ float y() const { return e[1]; }
        __HD__ float z() const { return e[2]; }

        __HD__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
        __HD__ float operator[](int i) const { return e[i]; }
        __HD__ float& operator[](int i) { return e[i]; }

        __HD__ vec3& operator+=(const vec3 &v) {
            e[0] += v[0];
            e[1] += v[1];
            e[2] += v[2];
            return *this;
        }

        __HD__ vec3& operator*=(const float t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }

        __HD__ vec3& operator/=(const float t) {
            return *this *= 1/t;
        }

        __HD__ float length() const {
            return sqrt(length_squared());
        }

        __HD__ float length_squared() const {
            return (e[0] * e[0]) + (e[1] * e[1]) + (e[2] * e[2]);
        }

        __device__ bool near_zero() const {
            // Return true if the vector is close to zero in all dimensions.
            const float s = 1e-8;
            return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
        }

        __host__ static vec3 random() {
            return vec3(random_float(), random_float(), random_float());
        }

        __host__ static vec3 random(float min, float max) {
            return vec3(random_float(min, max),
                        random_float(min, max),
                        random_float(min, max));
        }

        template <typename curandState>
        __device__ static vec3 random(curandState *state) {
            return vec3(random_float(state), random_float(state), random_float(state));
        }

        template <typename curandState>
        __device__ static vec3 random(curandState *state, float min, float max) {
            return vec3(random_float(state, min, max),
                        random_float(state, min, max),
                        random_float(state, min, max));
        }

    private:
        float e[3];
};

// Type aliases for vec3
using point3 = vec3;    // 3D point
using color = vec3;     // RGB color

// vec3 Utility Functions

inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v[0] << ' ' << v[1] << ' ' << v[2];
}

__HD__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u[0] + v[0], u[1] + v[1], u[2] + v[2]);
}

__HD__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u[0] - v[0], u[1] - v[1], u[2] - v[2]);
}

__HD__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u[0] * v[0], u[1] * v[1], u[2] * v[2]);
}

__HD__ inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t * v[0], t * v[1], t * v[2]);
}

__HD__ inline vec3 operator*(const vec3 &v, float t) {
    return t * v;
}

__HD__ inline vec3 operator/(const vec3 &v, float t) {
    return 1/t * v;
}

__HD__ inline float dot(const vec3 &u, const vec3 &v) {
    return (u[0] * v[0]) + (u[1] * v[1]) + (u[2] * v[2]);
}

__HD__ inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0]);
}

__HD__ inline vec3 unit_vector(const vec3 &v) {
    return v / v.length();
}

template <typename curandState>
__device__ inline vec3 random_in_unit_sphere(curandState *state) {
    while (true) {
        auto p = vec3::random(state, -1, 1);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

template <typename curandState>
__device__ inline vec3 random_unit_vector(curandState *state) {
    return unit_vector(random_in_unit_sphere(state));
}

template <typename curandState>
__device__ inline vec3 random_in_unit_disk(curandState *state) {
    while (true) {
        auto p = vec3(random_float(state, -1, 1), random_float(state, -1, 1), 0);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

template <typename curandState>
__device__ inline vec3 random_in_hemisphere(curandState *state, const vec3 &normal) {
    vec3 in_unit_sphere = random_in_unit_sphere(state);
    return dot(in_unit_sphere, normal) > 0      // In the same hemisphere as normal.
        ? in_unit_sphere : -in_unit_sphere;
}

__device__ inline vec3 reflect(const vec3 &v, const vec3 &n) {
    return v - 2 * dot(v, n) * n;
}

__device__ inline vec3 refract(const vec3 &uv, const vec3 &n, double etai_over_etat) {
    // Cast to float avoids need for constexpr.
    float cos_theta = fmin(dot(-uv, n), (float) 1.0);
    auto r_out_perp = etai_over_etat * (uv + n * cos_theta);
    auto r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#ifndef __CUDACC__
    #undef __host__
    #undef __device__
#endif

#undef __HD__

#endif
