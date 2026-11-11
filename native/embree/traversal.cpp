#include "raystrack/traversal.hpp"

#include <embree4/rtcore.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace raystrack {
namespace {

using Device = std::unique_ptr<RTCDeviceTy, decltype(&rtcReleaseDevice)>;
using Scene = std::unique_ptr<RTCSceneTy, decltype(&rtcReleaseScene)>;
using Geometry = std::unique_ptr<RTCGeometryTy, decltype(&rtcReleaseGeometry)>;

void check_error(RTCDevice device, const char* operation) {
    const auto error = rtcGetDeviceError(device);
    if (error != RTC_ERROR_NONE) {
        throw std::runtime_error(std::string("Embree ") + operation + ": " + rtcGetErrorString(error));
    }
}

RTCRay embree_ray(const Ray& ray) {
    RTCRay result{};
    result.org_x = ray.origin.x;
    result.org_y = ray.origin.y;
    result.org_z = ray.origin.z;
    result.dir_x = ray.direction.x;
    result.dir_y = ray.direction.y;
    result.dir_z = ray.direction.z;
    result.tnear = ray.t_min;
    result.tfar = ray.t_max;
    result.mask = ~0u;
    return result;
}

Vec3 normal(const RTCHit& hit) {
    const double length = std::hypot(
        static_cast<double>(hit.Ng_x), static_cast<double>(hit.Ng_y), static_cast<double>(hit.Ng_z));
    if (length == 0.0) {
        return {};
    }
    return {static_cast<float>(hit.Ng_x / length),
            static_cast<float>(hit.Ng_y / length),
            static_cast<float>(hit.Ng_z / length)};
}

Vec3 hit_point(const SurfaceGeometry& surface, const RTCHit& hit) {
    const auto triangle = surface.triangles[hit.primID];
    const auto a = surface.vertices[triangle.a];
    const auto b = surface.vertices[triangle.b];
    const auto c = surface.vertices[triangle.c];
    const auto coordinate = [&](double base, double edge_u, double edge_v) {
        return static_cast<float>(base + hit.u * (edge_u - base) + hit.v * (edge_v - base));
    };
    return {coordinate(a.x, b.x, c.x), coordinate(a.y, b.y, c.y), coordinate(a.z, b.z, c.z)};
}

class EmbreeTraversal final : public Traversal {
public:
    explicit EmbreeTraversal(std::span<const SurfaceGeometry> surfaces)
        : device_(rtcNewDevice(nullptr), rtcReleaseDevice), scene_(nullptr, rtcReleaseScene) {
        if (!device_) {
            check_error(nullptr, "device creation");
            throw std::runtime_error("Embree returned a null device");
        }
        scene_.reset(rtcNewScene(device_.get()));
        check_error(device_.get(), "scene creation");
        if (!scene_) {
            throw std::runtime_error("Embree returned a null scene");
        }
        if (surfaces.size() > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("Too many surfaces for Embree hit IDs");
        }
        surfaces_.resize(surfaces.size());
        vertex_counts_.reserve(surfaces.size());
        rtcSetSceneFlags(scene_.get(), RTC_SCENE_FLAG_ROBUST | RTC_SCENE_FLAG_DYNAMIC);
        // LOW enables independent per-geometry acceleration and its REFIT hint.
        rtcSetSceneBuildQuality(scene_.get(), RTC_BUILD_QUALITY_LOW);
        for (std::size_t i = 0; i < surfaces.size(); ++i) {
            validate(surfaces[i]);
            vertex_counts_.push_back(surfaces[i].vertices.size());
            if (!surfaces[i].triangles.empty()) {
                attach(surfaces[i], static_cast<unsigned int>(i));
            }
        }
        rtcCommitScene(scene_.get());
        check_error(device_.get(), "scene commit");
    }

    bool update(std::span<const SurfaceGeometry> surfaces) override {
        if (surfaces.size() != surfaces_.size()) return false;
        // Check every surface before modifying a committed buffer. Equal counts
        // alone do not imply fixed topology (connectivity/winding may change).
        for (std::size_t i = 0; i < surfaces.size(); ++i) {
            validate(surfaces[i]);
            if (surfaces[i].vertices.size() != vertex_counts_[i]
                || surfaces[i].triangles.size() != surfaces_[i].triangles.size()) return false;
            if (!surfaces[i].triangles.empty()
                && std::memcmp(surfaces[i].triangles.data(), surfaces_[i].triangles.data(),
                               surfaces[i].triangles.size_bytes()) != 0) return false;
        }
        // Periodically rebuild to bound degradation after large deformations.
        // This is a conservative refresh policy, not a measured BVH quality metric.
        if (refits_ >= 32) return false;
        bool changed = false;
        for (std::size_t i = 0; i < surfaces.size(); ++i) {
            if (surfaces[i].triangles.empty()) continue;
            const auto bytes = surfaces[i].vertices.size_bytes();
            if (std::memcmp(surfaces[i].vertices.data(), surfaces_[i].vertices.data(), bytes) == 0) continue;
            const auto geometry = rtcGetGeometry(scene_.get(), static_cast<unsigned int>(i));
            check_error(device_.get(), "geometry lookup");
            auto* vertices = rtcGetGeometryBufferData(geometry, RTC_BUFFER_TYPE_VERTEX, 0);
            check_error(device_.get(), "vertex buffer lookup");
            if (!vertices) throw std::runtime_error("Embree returned a null vertex buffer");
            std::memcpy(vertices, surfaces[i].vertices.data(), bytes);
            rtcUpdateGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0);
            rtcCommitGeometry(geometry);
            check_error(device_.get(), "geometry update");
            changed = true;
        }
        if (changed) {
            rtcCommitScene(scene_.get());
            check_error(device_.get(), "scene update");
            ++refits_;
        }
        return true;
    }

    void intersect(std::span<const Ray> rays, std::span<Hit> hits) const override {
        if (rays.size() != hits.size()) {
            throw std::invalid_argument("Ray and hit batch lengths differ");
        }
        for (std::size_t i = 0; i < rays.size(); ++i) {
            RTCRayHit query{};
            query.ray = embree_ray(rays[i]);
            query.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            rtcIntersect1(scene_.get(), &query, nullptr);
            if (query.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
                hits[i] = Hit{};
            } else {
                hits[i] = {static_cast<std::int32_t>(query.hit.geomID),
                           static_cast<std::int32_t>(query.hit.primID),
                           query.ray.tfar, query.hit.u, query.hit.v, normal(query.hit),
                           hit_point(surfaces_[query.hit.geomID], query.hit)};
            }
        }
    }

    void occluded(std::span<const Ray> rays, std::span<std::uint8_t> mask) const override {
        if (rays.size() != mask.size()) {
            throw std::invalid_argument("Ray and occlusion batch lengths differ");
        }
        for (std::size_t i = 0; i < rays.size(); ++i) {
            auto query = embree_ray(rays[i]);
            rtcOccluded1(scene_.get(), &query, nullptr);
            mask[i] = query.tfar < 0.0f;
        }
    }

private:
    static void validate(const SurfaceGeometry& surface) {
        if (surface.triangles.size() > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("Too many triangles for Embree hit IDs");
        }
        // Embree silently omits primitives outside this coordinate range.
        constexpr float max_coordinate = 1.844e18f;
        for (const auto& vertex : surface.vertices) {
            if (!(std::abs(vertex.x) <= max_coordinate && std::abs(vertex.y) <= max_coordinate
                  && std::abs(vertex.z) <= max_coordinate)) {
                throw std::invalid_argument("Vertex outside Embree's coordinate range");
            }
        }
        const auto vertex_count = surface.vertices.size();
        for (const auto& triangle : surface.triangles) {
            if (triangle.a >= vertex_count || triangle.b >= vertex_count || triangle.c >= vertex_count) {
                throw std::invalid_argument("Embree triangle index is out of bounds");
            }
        }
    }

    void attach(const SurfaceGeometry& surface, unsigned int id) {
        const Geometry geometry(rtcNewGeometry(device_.get(), RTC_GEOMETRY_TYPE_TRIANGLE),
                                rtcReleaseGeometry);
        check_error(device_.get(), "geometry creation");
        if (!geometry) {
            throw std::runtime_error("Embree returned a null geometry");
        }
        rtcSetGeometryBuildQuality(geometry.get(), RTC_BUILD_QUALITY_REFIT);
        const auto copy = [&](RTCBufferType type, RTCFormat format, auto data) {
            auto* buffer = rtcSetNewGeometryBuffer(geometry.get(), type, 0, format,
                                                  sizeof(data.front()), data.size());
            check_error(device_.get(), "geometry buffer allocation");
            if (!buffer) {
                throw std::runtime_error("Embree returned a null geometry buffer");
            }
            std::memcpy(buffer, data.data(), data.size_bytes());
            using Element = typename decltype(data)::element_type;
            return std::span<Element>(static_cast<Element*>(buffer), data.size());
        };
        static_assert(sizeof(Vec3) == 3 * sizeof(float));
        static_assert(sizeof(Triangle) == 3 * sizeof(std::uint32_t));
        const auto vertices = copy(RTC_BUFFER_TYPE_VERTEX, RTC_FORMAT_FLOAT3, surface.vertices);
        const auto triangles = copy(RTC_BUFFER_TYPE_INDEX, RTC_FORMAT_UINT3, surface.triangles);
        rtcCommitGeometry(geometry.get());
        check_error(device_.get(), "geometry commit");
        rtcAttachGeometryByID(scene_.get(), geometry.get(), id);
        check_error(device_.get(), "geometry attachment");
        // The committed scene retains the geometry and its allocated buffers.
        surfaces_[id] = {vertices, triangles};
    }

    Device device_;
    Scene scene_;
    std::vector<SurfaceGeometry> surfaces_;
    std::vector<std::size_t> vertex_counts_;
    unsigned int refits_ = 0;
};

}

std::unique_ptr<Traversal> make_embree(std::span<const SurfaceGeometry> surfaces) {
    return std::make_unique<EmbreeTraversal>(surfaces);
}

}
