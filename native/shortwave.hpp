#pragma once

#include <raystrack/traversal.hpp>
#include <array>

namespace raystrack {

using RGB = std::array<double, 3>;

struct Sensor { Vec3 point, normal; };
struct OpticalMaterial {
    RGB front, back, transmission;
    const double* klems;
    const double* reverse_cdf;
    Vec3 tangent, alternate;
    // ABI 4: finite diffuse/Klems (0), clear plate (1), solid boundary (2).
    std::uint32_t kind;
    double refractive_index;
    const Vec3* delta_normals;
    std::size_t delta_normal_count;
    RGB absorption; // Beer extinction in inverse metres, kind 2 only.
};

struct Lighting {
    const Vec3* sun;
    const RGB* dni;
    // One RGB row per timestep for uniform sky; one row per patch otherwise.
    const RGB* sky;
    const std::int32_t* ring_offsets;
    std::size_t rings, sky_stride;
    double ring_height, north_degrees;
};

void shortwave(const Traversal& traversal, std::span<const Sensor> sensors,
               const Lighting& light, std::size_t steps,
               std::span<const OpticalMaterial> materials, std::uint32_t samples,
               std::uint64_t seed, std::uint32_t bounces, float epsilon, RGB* result);

}
