#include "raystrack/traversal.hpp"
#include <embree4/rtcore.h>
#include <algorithm>
#include <array>
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

void check(RTCDevice device) {
    const auto error = rtcGetDeviceError(device);
    if (error != RTC_ERROR_NONE)
        throw std::runtime_error(std::string("Embree instances: ") + rtcGetErrorString(error));
}

RTCRay ray(const Ray& source) {
    RTCRay r{};
    r.org_x = source.origin.x; r.org_y = source.origin.y; r.org_z = source.origin.z;
    r.dir_x = source.direction.x; r.dir_y = source.direction.y; r.dir_z = source.direction.z;
    r.tnear = source.t_min; r.tfar = source.t_max; r.mask = ~0u;
    return r;
}

struct Prototype {
    Scene scene{nullptr, rtcReleaseScene};
    Geometry geometry{nullptr, rtcReleaseGeometry};
    std::vector<Vec3> vertices;
    std::vector<Triangle> triangles;
};

void validate(std::span<const SurfaceGeometry> prototypes,
              std::span<const GeometryInstance> instances) {
    if (instances.size() > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument("Too many Embree instance IDs");
    std::vector<std::array<double, 6>> bounds;
    bounds.reserve(prototypes.size());
    for (const auto& p : prototypes) {
        std::array<double, 6> bound{};
        if (!p.vertices.empty()) bound = {p.vertices[0].x, p.vertices[0].y, p.vertices[0].z,
                                          p.vertices[0].x, p.vertices[0].y, p.vertices[0].z};
        if (p.triangles.size() > std::numeric_limits<std::int32_t>::max())
            throw std::invalid_argument("Too many Embree primitive IDs");
        for (const auto& v : p.vertices) {
            if (!(std::abs(v.x) <= 1.844e18f && std::abs(v.y) <= 1.844e18f && std::abs(v.z) <= 1.844e18f))
                throw std::invalid_argument("Vertex outside Embree's coordinate range");
            const double xyz[] = {v.x, v.y, v.z};
            for (int k = 0; k < 3; ++k) {
                bound[k] = std::min(bound[k], xyz[k]);
                bound[k+3] = std::max(bound[k+3], xyz[k]);
            }
        }
        for (const auto& t : p.triangles)
            if (t.a >= p.vertices.size() || t.b >= p.vertices.size() || t.c >= p.vertices.size())
                throw std::invalid_argument("Embree triangle index is out of bounds");
        bounds.push_back(bound);
    }
    for (const auto& instance : instances) {
        if (instance.prototype >= prototypes.size())
            throw std::invalid_argument("Embree instance prototype is out of bounds");
        const auto* m = instance.transform;
        for (int i = 0; i < 12; ++i)
            if (!std::isfinite(m[i])) throw std::invalid_argument("Instance transform must be finite");
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double product = 0;
                for (int k = 0; k < 3; ++k) product += double(m[4*k+i]) * m[4*k+j];
                if (std::abs(product - (i == j)) > 3e-6)
                    throw std::invalid_argument("Instance transform must be proper rigid");
            }
        const double det = double(m[0])*(double(m[5])*m[10]-double(m[6])*m[9])
                         - double(m[1])*(double(m[4])*m[10]-double(m[6])*m[8])
                         + double(m[2])*(double(m[4])*m[9]-double(m[5])*m[8]);
        if (std::abs(det - 1) > 3e-6) throw std::invalid_argument("Instance transform must be proper rigid");
        // Validate transformed bounds without preparing per-instance vertices.
        const auto& p = prototypes[instance.prototype];
        if (!p.vertices.empty()) {
            const auto& b = bounds[instance.prototype];
            for (int i = 0; i < 3; ++i) {
                double center = m[4*i+3], extent = 0;
                for (int j = 0; j < 3; ++j) {
                    center += m[4*i+j] * ((b[j]+b[j+3])*0.5);
                    extent += std::abs(m[4*i+j]) * ((b[j+3]-b[j])*0.5);
                }
                if (std::abs(center) + extent > 1.844e18)
                    throw std::invalid_argument("Instance outside Embree's coordinate range");
            }
        }
    }
}

class Instances final : public Traversal {
public:
    Instances(std::span<const SurfaceGeometry> prototypes, std::span<const GeometryInstance> instances)
        : device_(rtcNewDevice(nullptr), rtcReleaseDevice), scene_(nullptr, rtcReleaseScene) {
        if (!device_) throw std::runtime_error("Embree instance device creation failed");
        validate(prototypes, instances);
        scene_.reset(rtcNewScene(device_.get()));
        rtcSetSceneFlags(scene_.get(), RTC_SCENE_FLAG_DYNAMIC | RTC_SCENE_FLAG_ROBUST);
        rtcSetSceneBuildQuality(scene_.get(), RTC_BUILD_QUALITY_LOW);
        for (const auto& input : prototypes) {
            Prototype p;
            p.vertices.assign(input.vertices.begin(), input.vertices.end());
            p.triangles.assign(input.triangles.begin(), input.triangles.end());
            p.scene.reset(rtcNewScene(device_.get()));
            rtcSetSceneFlags(p.scene.get(), RTC_SCENE_FLAG_DYNAMIC | RTC_SCENE_FLAG_ROBUST);
            rtcSetSceneBuildQuality(p.scene.get(), RTC_BUILD_QUALITY_LOW);
            if (!p.triangles.empty()) {
                p.geometry.reset(rtcNewGeometry(device_.get(), RTC_GEOMETRY_TYPE_TRIANGLE));
                rtcSetGeometryBuildQuality(p.geometry.get(), RTC_BUILD_QUALITY_REFIT);
                auto* vertices = rtcSetNewGeometryBuffer(p.geometry.get(), RTC_BUFFER_TYPE_VERTEX, 0,
                    RTC_FORMAT_FLOAT3, sizeof(Vec3), p.vertices.size());
                auto* triangles = rtcSetNewGeometryBuffer(p.geometry.get(), RTC_BUFFER_TYPE_INDEX, 0,
                    RTC_FORMAT_UINT3, sizeof(Triangle), p.triangles.size());
                check(device_.get());
                if (!vertices || !triangles) throw std::runtime_error("Embree prototype buffer allocation failed");
                std::memcpy(vertices, p.vertices.data(), input.vertices.size_bytes());
                std::memcpy(triangles, p.triangles.data(), input.triangles.size_bytes());
                rtcCommitGeometry(p.geometry.get());
                rtcAttachGeometryByID(p.scene.get(), p.geometry.get(), 0);
            }
            rtcCommitScene(p.scene.get());
            check(device_.get());
            prototypes_.push_back(std::move(p));
            ++stats_.prototype_builds;
        }
        instances_.assign(instances.begin(), instances.end());
        for (std::size_t i = 0; i < instances.size(); ++i) {
            const auto& instance = instances[i];
            Geometry g(rtcNewGeometry(device_.get(), RTC_GEOMETRY_TYPE_INSTANCE), rtcReleaseGeometry);
            rtcSetGeometryInstancedScene(g.get(), prototypes_[instance.prototype].scene.get());
            rtcSetGeometryTransform(g.get(), 0, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, instance.transform);
            rtcCommitGeometry(g.get());
            rtcAttachGeometryByID(scene_.get(), g.get(), static_cast<unsigned>(i));
            check(device_.get());
        }
        rtcCommitScene(scene_.get()); check(device_.get()); ++stats_.scene_commits;
    }

    bool update_instances(std::span<const SurfaceGeometry> prototypes,
                          std::span<const GeometryInstance> instances) override {
        validate(prototypes, instances); // Complete preflight before touching committed buffers.
        if (prototypes.size() != prototypes_.size() || instances.size() != instances_.size()) return false;
        for (std::size_t i = 0; i < prototypes.size(); ++i) {
            const auto& p = prototypes_[i]; const auto& input = prototypes[i];
            if (p.vertices.size() != input.vertices.size() || p.triangles.size() != input.triangles.size()
                || (!p.triangles.empty() && std::memcmp(p.triangles.data(), input.triangles.data(), input.triangles.size_bytes())))
                return false;
        }
        for (std::size_t i = 0; i < instances.size(); ++i)
            if (instances[i].prototype != instances_[i].prototype) return false;
        bool changed = false;
        for (std::size_t i = 0; i < prototypes.size(); ++i) {
            auto& p = prototypes_[i]; const auto& input = prototypes[i];
            if (p.vertices.empty() || !std::memcmp(p.vertices.data(), input.vertices.data(), input.vertices.size_bytes())) continue;
            if (refits_ >= 32) return false;
            changed = true;
        }
        // Above loop also checks periodic refresh before any mutation.
        for (std::size_t i = 0; i < prototypes.size(); ++i) {
            auto& p = prototypes_[i]; const auto& input = prototypes[i];
            if (p.vertices.empty() || !std::memcmp(p.vertices.data(), input.vertices.data(), input.vertices.size_bytes())) continue;
            p.vertices.assign(input.vertices.begin(), input.vertices.end());
            if (p.geometry) {
                auto* buffer = rtcGetGeometryBufferData(p.geometry.get(), RTC_BUFFER_TYPE_VERTEX, 0);
                std::memcpy(buffer, p.vertices.data(), input.vertices.size_bytes());
                rtcUpdateGeometryBuffer(p.geometry.get(), RTC_BUFFER_TYPE_VERTEX, 0);
                rtcCommitGeometry(p.geometry.get()); rtcCommitScene(p.scene.get()); check(device_.get());
                ++stats_.prototype_refits;
            }
        }
        const bool refitted = changed;
        for (std::size_t i = 0; i < instances.size(); ++i) {
            const bool moved = std::memcmp(instances[i].transform, instances_[i].transform, sizeof(instances[i].transform));
            if (!moved && !refitted) continue;
            const auto g = rtcGetGeometry(scene_.get(), static_cast<unsigned>(i));
            if (moved) {
                rtcSetGeometryTransform(g, 0, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, instances[i].transform);
                ++stats_.transform_commits;
            }
            rtcCommitGeometry(g); check(device_.get()); changed = true;
        }
        if (changed) {
            rtcCommitScene(scene_.get()); check(device_.get()); ++stats_.scene_commits;
            if (refitted) ++refits_;
        }
        instances_.assign(instances.begin(), instances.end());
        return true;
    }

    InstanceStatistics instance_statistics() const override { return stats_; }

    void intersect(std::span<const Ray> rays, std::span<Hit> hits) const override {
        if (rays.size() != hits.size()) throw std::invalid_argument("Ray and hit lengths differ");
        for (std::size_t i = 0; i < rays.size(); ++i) {
            RTCRayHit query{}; query.ray = ray(rays[i]); query.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            for (auto& id : query.hit.instID) id = RTC_INVALID_GEOMETRY_ID;
            rtcIntersect1(scene_.get(), &query, nullptr);
            if (query.hit.geomID == RTC_INVALID_GEOMETRY_ID) { hits[i] = {}; continue; }
            const auto slot = query.hit.instID[0];
            const auto& instance = instances_[slot];
            const auto& p = prototypes_[instance.prototype];
            const auto t = p.triangles[query.hit.primID];
            const auto a = p.vertices[t.a], b = p.vertices[t.b], c = p.vertices[t.c];
            const double u = query.hit.u, v = query.hit.v;
            const double point[] = {a.x+u*(double(b.x)-a.x)+v*(double(c.x)-a.x),
                                    a.y+u*(double(b.y)-a.y)+v*(double(c.y)-a.y),
                                    a.z+u*(double(b.z)-a.z)+v*(double(c.z)-a.z)};
            const double n[] = {(double(b.y)-a.y)*(double(c.z)-a.z)-(double(b.z)-a.z)*(double(c.y)-a.y),
                                (double(b.z)-a.z)*(double(c.x)-a.x)-(double(b.x)-a.x)*(double(c.z)-a.z),
                                (double(b.x)-a.x)*(double(c.y)-a.y)-(double(b.y)-a.y)*(double(c.x)-a.x)};
            const auto* m = instance.transform;
            float world[3], normal[3];
            const auto length = std::hypot(n[0], n[1], n[2]);
            for (int k = 0; k < 3; ++k) {
                world[k] = static_cast<float>(m[4*k]*point[0]+m[4*k+1]*point[1]+m[4*k+2]*point[2]+m[4*k+3]);
                normal[k] = length ? static_cast<float>((m[4*k]*n[0]+m[4*k+1]*n[1]+m[4*k+2]*n[2])/length) : 0;
            }
            hits[i] = {static_cast<std::int32_t>(slot), static_cast<std::int32_t>(query.hit.primID),
                       query.ray.tfar, query.hit.u, query.hit.v, {normal[0],normal[1],normal[2]}, {world[0],world[1],world[2]}};
        }
    }

    void occluded(std::span<const Ray> rays, std::span<std::uint8_t> mask) const override {
        if (rays.size() != mask.size()) throw std::invalid_argument("Ray and occlusion lengths differ");
        for (std::size_t i = 0; i < rays.size(); ++i) {
            auto query = ray(rays[i]); rtcOccluded1(scene_.get(), &query, nullptr); mask[i] = query.tfar < 0;
        }
    }
private:
    Device device_;
    std::vector<Prototype> prototypes_;
    std::vector<GeometryInstance> instances_;
    Scene scene_;
    InstanceStatistics stats_;
    unsigned refits_ = 0;
};
}

std::unique_ptr<Traversal> make_embree_instances(std::span<const SurfaceGeometry> prototypes,
                                               std::span<const GeometryInstance> instances) {
    return std::make_unique<Instances>(prototypes, instances);
}
}
