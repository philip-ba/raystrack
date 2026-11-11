#pragma once

#include "shortwave.hpp"
#include "vector.hpp"
#include <algorithm>
#include <cmath>

namespace raystrack {

// A nonabsorbing, incoherent parallel plate in air. Sum internal reflections
// separately for each polarization before averaging; there is no eta^2 factor
// because both external media have the same refractive index.
inline double plate_transmission(double cosine, double index) {
    if (index == 1) return 1;
    cosine = std::clamp(std::abs(cosine), 0.0, 1.0);
    const double eta = 1 / index;
    const double inside = std::sqrt(1 - (1 - cosine*cosine) * eta*eta);
    const double s = (eta*cosine - inside) / (eta*cosine + inside);
    const double p = (cosine - eta*inside) / (cosine + eta*inside);
    return 0.5 * ((1 - s*s)/(1 + s*s) + (1 - p*p)/(1 + p*p));
}

inline Vec3 specular_reflection(Vec3 direction, Vec3 normal) {
    return normalize(direction - normal * (2 * dot(direction, normal)));
}

struct DielectricEvent {
    double reflectance, eta;
    Vec3 transmission;
};

// Reverse ray direction, normal facing its current medium. Radiance throughput
// on transmission is (1-R)*eta^2, while Fresnel sampling mass is (1-R).
inline DielectricEvent dielectric(Vec3 direction, Vec3 normal, double index, bool entering) {
    if (index == 1) return {0, 1, direction};
    const double eta = entering ? 1/index : index;
    const double normal_length = std::hypot(normal.x, normal.y, normal.z);
    const double nx = normal.x/normal_length, ny = normal.y/normal_length, nz = normal.z/normal_length;
    const double length = std::hypot(direction.x, direction.y, direction.z);
    const double dx = direction.x/length, dy = direction.y/length, dz = direction.z/length;
    const double cosine = std::clamp(-(dx*nx+dy*ny+dz*nz), 0.0, 1.0);
    const double tx = dx+cosine*nx, ty = dy+cosine*ny, tz = dz+cosine*nz;
    // Tangential length retains Snell information when float direction cosine
    // rounds to one inside a high-index medium.
    const double sine_squared = eta*eta*(tx*tx+ty*ty+tz*tz);
    if (sine_squared >= 1) return {1, eta, {}};
    const double inside = std::sqrt(std::max(0.0, 1-sine_squared));
    const double s = index == 1 ? 0 : (eta*cosine-inside)/(eta*cosine+inside);
    const double p = index == 1 ? 0 : (cosine-eta*inside)/(cosine+eta*inside);
    const Vec3 transmitted{
        static_cast<float>(eta*tx-inside*nx),
        static_cast<float>(eta*ty-inside*ny),
        static_cast<float>(eta*tz-inside*nz)};
    return {.5*(s*s+p*p), eta, normalize(transmitted)};
}

}
