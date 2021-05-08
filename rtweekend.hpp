#ifndef RTWEEKEND_HPP
#define RTWEEKEND_HPP

#include <cmath>
#include <limits>

#ifndef __CUDACC__
    #define __host__
    #define __device__
#endif

// Usings

using std::sqrt;

// Constants

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385;

// Utility Functions

inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

__host__ inline float random_float() {
    return rand() / (RAND_MAX + 1.0);
}

__host__ inline float random_float(float min, float max) {
    return min + (max-min) * random_float();
}

template <typename curandState>
__device__ inline float random_float(curandState *local_state) {
    // Returns a random real in (0, 1].
    return curand_uniform(local_state);
}

template <typename curandState>
__device__ inline float random_float(curandState *local_state, float min, float max) {
    // Returns a random real in (min, max].
    return min + (max-min) * curand_uniform(local_state);
}

inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

#ifndef __CUDACC__
    #undef __host__
    #undef __device__
#endif

// Common Headers

#endif
