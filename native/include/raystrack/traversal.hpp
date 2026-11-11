#pragma once

#include <cstdint>
#include <memory>
#include <span>

namespace raystrack {

struct Vec3 { float x, y, z; };
struct Triangle { std::uint32_t a, b, c; };

struct SurfaceGeometry {
    std::span<const Vec3> vertices;
    std::span<const Triangle> triangles;
};

// Absolute local-to-world proper rigid transform, row-major 3x4. Array slots
// identify surfaces; BLAS geometry and primitive IDs stay local.
struct GeometryInstance {
    std::uint32_t prototype;
    float transform[12];
};

struct InstanceStatistics {
    std::uint64_t prototype_builds = 0, prototype_refits = 0;
    std::uint64_t transform_commits = 0, scene_commits = 0;
};

struct Ray {
    Vec3 origin, direction;
    float t_min, t_max;
};

struct Hit {
    std::int32_t surface = -1, primitive = -1;
    float distance = 0, u = 0, v = 0;
    Vec3 normal{};
    Vec3 point{};
};

class Traversal {
public:
    virtual ~Traversal() = default;
    // Called only while tracing is idle. False requests a fresh traversal;
    // an unsupported update must leave this snapshot unchanged.
    virtual bool update(std::span<const SurfaceGeometry>) { return false; }
    virtual bool update_instances(std::span<const SurfaceGeometry>,
                                  std::span<const GeometryInstance>) { return false; }
    virtual InstanceStatistics instance_statistics() const { return {}; }
    virtual void intersect(std::span<const Ray> rays, std::span<Hit> hits) const = 0;
    virtual void occluded(std::span<const Ray> rays, std::span<std::uint8_t> mask) const = 0;
};

std::unique_ptr<Traversal> make_builtin(std::span<const SurfaceGeometry> surfaces);
std::unique_ptr<Traversal> make_embree(std::span<const SurfaceGeometry> surfaces);
std::unique_ptr<Traversal> make_embree_instances(std::span<const SurfaceGeometry> prototypes,
                                               std::span<const GeometryInstance> instances);
std::unique_ptr<Traversal> make_vulkan(std::span<const SurfaceGeometry> surfaces);

}
