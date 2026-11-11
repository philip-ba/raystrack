#include "raystrack/traversal.hpp"

#include <array>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* description) {
    if (!condition) {
        throw std::runtime_error(description);
    }
}

bool near(float actual, float expected) {
    return std::abs(actual - expected) <= 1e-5f;
}

void probe() {
    using namespace raystrack;
    constexpr std::array upper{
        Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{0, 1, 0},
        Vec3{2, 0, 0}, Vec3{3, 0, 0}, Vec3{2, 1, 0},
    };
    constexpr std::array lower{Vec3{0, 0, -1}, Vec3{1, 0, -1}, Vec3{0, 1, -1}};
    constexpr std::array upper_triangles{Triangle{0, 1, 2}, Triangle{3, 4, 5}};
    constexpr std::array lower_triangles{Triangle{0, 2, 1}};
    const std::array geometry{
        SurfaceGeometry{upper, upper_triangles}, SurfaceGeometry{lower, lower_triangles},
    };
    auto scene = make_vulkan(geometry);
    constexpr float infinity = std::numeric_limits<float>::infinity();
    constexpr std::array rays{
        Ray{{0.2f, 0.3f, 1}, {0, 0, -1}, 0, infinity},
        Ray{{0.2f, 0.3f, 1}, {0, 0, -1}, 1.1f, infinity},
        Ray{{0.2f, 0.3f, 1}, {0, 0, -1}, 0, 0.9f},
        Ray{{2.25f, 0.25f, -1}, {0, 0, 1}, 0, infinity},
        Ray{{4, 4, 1}, {0, 0, -1}, 0, infinity},
    };
    std::array<Hit, rays.size()> hits;
    scene->intersect(rays, hits);
    require(hits[0].surface == 0 && hits[0].primitive == 0 && near(hits[0].distance, 1),
            "closest hit returned the wrong surface, primitive, or distance");
    require(near(hits[0].u, 0.2f) && near(hits[0].v, 0.3f) && near(hits[0].normal.z, 1),
            "closest hit returned the wrong barycentrics or winding normal");
    require(near(hits[0].point.x, 0.2f) && near(hits[0].point.y, 0.3f) && hits[0].point.z == 0,
            "closest hit returned the wrong position on the triangle");
    require(hits[1].surface == 1 && hits[1].primitive == 0 && near(hits[1].distance, 2)
            && near(hits[1].normal.z, -1), "t_min excluded the wrong primitive");
    require(hits[2].surface == -1 && hits[4].surface == -1, "miss returned a surface");
    require(hits[3].surface == 0 && hits[3].primitive == 1 && near(hits[3].normal.z, 1),
            "back-face ray changed the geometric normal or missed its primitive");
    std::array<uint8_t, rays.size()> occlusion{};
    scene->occluded(rays, occlusion);
    require(occlusion == std::array<uint8_t, rays.size()>{1, 1, 0, 1, 0}, "wrong bounded occlusion results");

    constexpr size_t batch_size = 262145;
    const std::vector<Ray> batch(batch_size, rays[0]);
    std::vector<uint8_t> batch_results(batch_size);
    scene->occluded(batch, batch_results);
    for (const auto result : batch_results) {
        require(result == 1, "multi-dispatch batch dropped an occluded ray");
    }

    auto moved = upper;
    for (auto& vertex : moved) {
        vertex.x += 8;
    }
    const std::array edited_geometry{
        SurfaceGeometry{moved, upper_triangles}, SurfaceGeometry{lower, lower_triangles},
    };
    const auto edited = make_vulkan(edited_geometry);
    edited->intersect(rays, hits);
    require(hits[0].surface == 1 && near(hits[0].distance, 2), "rebuild did not see moved geometry");
    scene->intersect(rays, hits);
    require(hits[0].surface == 0 && near(hits[0].distance, 1), "rebuild changed the previous scene");

    require(scene->update(edited_geometry), "Vulkan did not retain its device across a geometry edit");
    std::array<Hit, rays.size()> reference;
    edited->intersect(rays, reference);
    scene->intersect(rays, hits);
    for (size_t index = 0; index < hits.size(); ++index) {
        require(hits[index].surface == reference[index].surface
                && hits[index].primitive == reference[index].primitive
                && near(hits[index].distance, reference[index].distance)
                && near(hits[index].normal.z, reference[index].normal.z),
                "updated traversal differs from a fresh build");
    }

    // Invalid edits must leave both host metadata and acceleration usable.
    constexpr std::array invalid_triangles{Triangle{0, 1, 99}};
    const std::array invalid_geometry{SurfaceGeometry{upper, invalid_triangles}};
    bool rejected = false;
    try {
        scene->update(invalid_geometry);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "out-of-range topology was not rejected");
    scene->intersect(rays, hits);
    require(hits[0].surface == 1 && near(hits[0].distance, 2), "failed update damaged the previous geometry");

    // Topology edits and changing surface counts retain the pipeline too.
    constexpr std::array reversed{Triangle{0, 2, 1}};
    const std::array topology{SurfaceGeometry{upper, reversed}};
    require(scene->update(topology), "topology update replaced the Vulkan device");
    scene->intersect(rays, hits);
    require(hits[0].surface == 0 && hits[0].primitive == 0 && near(hits[0].normal.z, -1),
            "topology update returned stale surface, primitive, or normal");
    require(hits[3].surface == -1, "topology update retained a removed triangle");
    require(scene->update({}), "empty update replaced the Vulkan device");
    scene->intersect(rays, hits);
    scene->occluded(rays, occlusion);
    for (size_t index = 0; index < hits.size(); ++index) {
        require(hits[index].surface == -1 && occlusion[index] == 0, "empty update retained geometry");
    }
    require(scene->update(geometry), "refilling empty scene replaced the Vulkan device");
    scene->intersect(rays, hits);
    require(hits[0].surface == 0 && near(hits[0].distance, 1), "refilled geometry returned a stale hit");

    auto initially_empty = make_vulkan({});
    require(initially_empty->update(geometry), "initially empty scene did not accept geometry");
    initially_empty->intersect(rays, hits);
    require(hits[0].surface == 0 && near(hits[0].distance, 1), "initially empty scene did not create a pipeline");
}

} // namespace

int main() {
    try {
        probe();
        std::cout << "Vulkan ray queries passed closest-hit, bounded occlusion, update/fresh parity, topology, rollback and empty-scene checks.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
