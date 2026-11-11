#include "shortwave.hpp"
#include "vector.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#define RS_EXPORT extern "C" __declspec(dllexport)
#else
#define RS_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

using namespace raystrack;
static_assert(sizeof(Vec3) == 3 * sizeof(float));
static_assert(sizeof(Triangle) == 3 * sizeof(std::uint32_t));
static_assert(sizeof(Sensor) == 6 * sizeof(float));

struct SurfaceInput {
    const Vec3* vertices;
    const Triangle* triangles;
    std::size_t vertex_count, triangle_count;
};

struct SceneHandle {
    std::unique_ptr<Traversal> traversal;
    std::size_t surfaces;
    int backend;
};

thread_local std::string error;

template<typename Function>
int guarded(Function&& operation) noexcept {
    try {
        operation();
        return 0;
    } catch (const std::exception& e) {
        error = e.what();
    } catch (...) {
        error = "unknown native error";
    }
    return 1;
}

std::unique_ptr<Traversal> create(int backend, std::span<const SurfaceGeometry> geometry) {
    switch (backend) {
    case 0:
        return make_builtin(geometry);
#ifdef RAYSTRACK_EMBREE
    case 1:
        return make_embree(geometry);
#endif
#ifdef RAYSTRACK_VULKAN
    case 2:
        return make_vulkan(geometry);
#endif
    default:
        throw std::runtime_error("requested native backend is absent from this build");
    }
}

}

RS_EXPORT unsigned rs_abi() noexcept { return 4; }

RS_EXPORT unsigned rs_backends() noexcept {
    return 1
#ifdef RAYSTRACK_EMBREE
        | 2
#endif
#ifdef RAYSTRACK_VULKAN
        | 4
#endif
        ;
}

RS_EXPORT const char* rs_error() noexcept { return error.c_str(); }

RS_EXPORT int rs_create(int backend, const SurfaceInput* input, std::size_t count,
                        SceneHandle** result) noexcept {
    return guarded([&] {
        std::vector<SurfaceGeometry> geometry;
        geometry.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            geometry.push_back({{input[i].vertices, input[i].vertex_count},
                                {input[i].triangles, input[i].triangle_count}});
        }
        *result = new SceneHandle{create(backend, geometry), count, backend};
    });
}

RS_EXPORT int rs_update(SceneHandle* handle, const SurfaceInput* input, std::size_t count,
                        int* retained) noexcept {
    return guarded([&] {
        std::vector<SurfaceGeometry> geometry;
        geometry.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            geometry.push_back({{input[i].vertices, input[i].vertex_count},
                                {input[i].triangles, input[i].triangle_count}});
        }
        const bool reused = handle->traversal->update(geometry);
        if (!reused) {
            // Construct before replacing so a failed rebuild retains ownership.
            auto replacement = create(handle->backend, geometry);
            handle->traversal = std::move(replacement);
        }
        handle->surfaces = count;
        if (retained) *retained = reused;
    });
}

RS_EXPORT void rs_destroy(SceneHandle* handle) noexcept { delete handle; }

RS_EXPORT int rs_shortwave(const SceneHandle* handle, const Sensor* sensors, std::size_t count,
                           const Lighting* light, std::size_t steps, const OpticalMaterial* materials,
                           std::uint32_t samples, std::uint64_t seed, std::uint32_t bounces,
                           float epsilon, RGB* result) noexcept {
    return guarded([&] {
        for (std::size_t i = 0; i < handle->surfaces; ++i) {
            const auto& material = materials[i];
            if (material.kind > 2) throw std::runtime_error("unsupported native material kind");
            if (material.kind == 1 && (!std::isfinite(material.refractive_index)
                                      || material.refractive_index < 1))
                throw std::runtime_error("clear plate refractive index must be finite and at least one");
            if (material.kind == 2) {
                if (!std::isfinite(material.refractive_index) || material.refractive_index < 1
                    || material.refractive_index > 10)
                    throw std::runtime_error("solid refractive index must be in [1, 10]");
                for (const auto value : material.absorption)
                    if (!std::isfinite(value) || value < 0)
                        throw std::runtime_error("solid absorption must be finite and nonnegative");
                for (std::size_t step = 0; step < steps; ++step)
                    for (const auto value : light->dni[step])
                        if (value != 0)
                            throw std::runtime_error("capability: SolidDielectric supports sky forcing only; DNI must be zero");
            }
        }
        shortwave(*handle->traversal, {sensors, count}, *light, steps,
                  {materials, handle->surfaces}, samples, seed, bounces, epsilon, result);
    });
}

RS_EXPORT int rs_longwave(const SceneHandle* handle, const Vec3* origins, const Vec3* directions,
                          std::size_t count, const std::uint8_t* active,
                          std::int32_t excluded, std::int32_t min_surface,
                          std::int32_t* surfaces, std::uint8_t* front,
                          std::uint8_t* blocked) noexcept {
    return guarded([&] {
        const float infinity = std::numeric_limits<float>::infinity();
        const float near = std::nextafter(1e-6f, infinity);
        std::vector<Ray> rays, retry;
        std::vector<std::size_t> slots, retry_slots;
        std::vector<Hit> hits;
        for (std::size_t i = 0; i < count; ++i) {
            rays.push_back({origins[i], directions[i], near, infinity});
            slots.push_back(i);
        }
        std::fill_n(surfaces, count, -1);
        std::fill_n(front, count, 0);
        std::fill_n(blocked, count, 0);
        while (!rays.empty()) {
            hits.resize(rays.size());
            handle->traversal->intersect(rays, hits);
            retry.clear();
            retry_slots.clear();
            for (std::size_t i = 0; i < rays.size(); ++i) {
                const auto& hit = hits[i];
                if (hit.surface < 0) continue;
                const auto slot = slots[i];
                if (hit.surface != excluded && active[hit.surface]) {
                    blocked[slot] = 1;
                    if (hit.surface >= min_surface) {
                        surfaces[slot] = hit.surface;
                        front[slot] = dot(directions[slot], hit.normal) < 0;
                        continue;
                    }
                }
                const auto next = std::nextafter(hit.distance, infinity);
                if (!(next > rays[i].t_min) || !std::isfinite(next)) continue;
                retry.push_back({origins[slot], directions[slot], next, infinity});
                retry_slots.push_back(slot);
            }
            rays.swap(retry);
            slots.swap(retry_slots);
        }
    });
}
