#include "scattering.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace raystrack;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() {
    try {
        for (const double index : {1., 1.52, 1e4, 1e6}) {
            const auto original = normalize(Vec3{1, 0, -1});
            const auto entry = dielectric(original, {0, 0, 1}, index, true);
            require(std::abs(index*entry.transmission.x-original.x) < 1e-6,
                    "entrance Snell tangent mismatch");
            const auto exit = dielectric(entry.transmission, {0, 0, 1}, index, false);
            require(std::abs(exit.transmission.x-original.x) < 2e-6
                    && std::abs(exit.transmission.z-original.z) < 2e-6,
                    "high-index parallel slab failed directional roundtrip");
            require(std::abs(entry.reflectance-exit.reflectance) < 1e-6,
                    "reciprocal interface Fresnel mismatch");
            require(std::abs(entry.eta*exit.eta-1) < 1e-12,
                    "air-to-air adjoint eta factor failed cancellation");
        }
        // Accepted API indices also retain direction when neither the normal
        // nor the refracted ray is aligned to a float coordinate axis.
        const auto normal = normalize(Vec3{.3f, .4f, .8660254f});
        const auto tangent_axis = normalize(cross(normal, Vec3{0, 1, 0}));
        for (const double index : {1.52, 10.}) {
            const auto original = normalize(normal * -.8f + tangent_axis * .6f);
            const auto entry = dielectric(original, normal, index, true);
            const auto exit = dielectric(entry.transmission, normal, index, false);
            require(std::hypot(exit.transmission.x-original.x,
                               exit.transmission.y-original.y,
                               exit.transmission.z-original.z) < 3e-6,
                    "rotated-normal parallel slab failed directional roundtrip");
        }
        const auto tir = dielectric(normalize(Vec3{.9f, 0, -.4f}), {0, 0, 1}, 1.52, false);
        require(tir.reflectance == 1, "total internal reflection was not detected");
        const auto tangent = dielectric({1, 0, 0}, {0, 0, 1}, 1, true);
        require(tangent.reflectance == 0 && tangent.transmission.x == 1,
                "matched-index grazing boundary changed direction or power");
        std::cout << "dielectric Snell/Fresnel/TIR checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
