#ifndef VEC3_HPP
#define VEC3_HPP

#include <cmath>
#include <iostream>

#ifdef __CUDACC__
    #define __HD__ __host__ __device__
#else
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

#undef __HD__

#endif
