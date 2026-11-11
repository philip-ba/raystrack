#include "vector.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace raystrack {
namespace {

using Point = std::array<float, 3>;
constexpr float infinity = std::numeric_limits<float>::infinity();

struct WideVec { double x, y, z; };

WideVec widen(Vec3 a) { return {a.x, a.y, a.z}; }
Vec3 narrow(WideVec a) {
    return {static_cast<float>(a.x), static_cast<float>(a.y), static_cast<float>(a.z)};
}
WideVec operator+(WideVec a, WideVec b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
WideVec operator-(WideVec a, WideVec b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
WideVec operator*(WideVec a, double b) { return {a.x * b, a.y * b, a.z * b}; }
double dot(WideVec a, WideVec b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
WideVec cross(WideVec a, WideVec b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}

struct Bounds {
    Point low{infinity, infinity, infinity}, high{-infinity, -infinity, -infinity};

    void include(Point p) {
        for (int i = 0; i < 3; ++i) {
            low[i] = std::min(low[i], p[i]);
            high[i] = std::max(high[i], p[i]);
        }
    }

    bool intersects(Point origin, Point direction, float near, float far) const {
        for (int i = 0; i < 3; ++i) {
            if (direction[i] == 0) {
                if (origin[i] < low[i] || origin[i] > high[i]) return false;
                continue;
            }
            float first = (low[i] - origin[i]) / direction[i];
            float last = (high[i] - origin[i]) / direction[i];
            if (first > last) std::swap(first, last);
            near = std::max(near, first);
            far = std::min(far, last);
            if (far < near) return false;
        }
        return true;
    }
};

struct Primitive {
    Bounds bounds;
    Point center;
    WideVec vertex, edge1, edge2;
    Vec3 normal;
    int surface, triangle;
};

struct Node {
    Bounds bounds;
    std::size_t begin = 0, end = 0, left = 0, right = 0;
};

class Builtin final : public Traversal {
public:
    explicit Builtin(std::span<const SurfaceGeometry> surfaces) {
        for (std::size_t surface = 0; surface < surfaces.size(); ++surface) {
            auto [vertices, triangles] = surfaces[surface];
            for (std::size_t face = 0; face < triangles.size(); ++face) {
                const auto [a, b, c] = triangles[face];
                const Vec3 p = vertices[a], q = vertices[b], r = vertices[c];
                const auto origin = widen(p);
                const WideVec edge1 = widen(q) - origin, edge2 = widen(r) - origin;
                const auto geometric = cross(edge1, edge2);
                const auto length = std::hypot(geometric.x, geometric.y, geometric.z);
                if (length == 0) continue;
                const Vec3 normal = narrow(geometric * (1.0 / length));
                Bounds box;
                for (Vec3 vertex : {p, q, r}) box.include({vertex.x, vertex.y, vertex.z});
                Point center{};
                for (int i = 0; i < 3; ++i) center[i] = box.low[i] * 0.5f + box.high[i] * 0.5f;
                primitives_.push_back({box, center, origin, edge1, edge2, normal,
                                       static_cast<int>(surface), static_cast<int>(face)});
            }
        }
        if (!primitives_.empty()) build(0, primitives_.size());
    }

    void intersect(std::span<const Ray> rays, std::span<Hit> hits) const override {
        for (std::size_t i = 0; i < rays.size(); ++i) hits[i] = trace(rays[i], false);
    }

    void occluded(std::span<const Ray> rays, std::span<std::uint8_t> mask) const override {
        for (std::size_t i = 0; i < rays.size(); ++i) mask[i] = trace(rays[i], true).surface >= 0;
    }

private:
    std::size_t build(std::size_t begin, std::size_t end) {
        Bounds bounds, centers;
        for (auto i = begin; i < end; ++i) {
            bounds.include(primitives_[i].bounds.low);
            bounds.include(primitives_[i].bounds.high);
            centers.include(primitives_[i].center);
        }
        const auto id = nodes_.size();
        nodes_.push_back({bounds, begin, end});
        if (end - begin <= 4) return id;
        int axis = 0;
        for (int i = 1; i < 3; ++i)
            if (centers.high[i] - centers.low[i] > centers.high[axis] - centers.low[axis]) axis = i;
        const auto mid = begin + (end - begin) / 2;
        std::nth_element(primitives_.begin() + begin, primitives_.begin() + mid,
                         primitives_.begin() + end, [axis](const Primitive& a, const Primitive& b) {
                             return a.center[axis] < b.center[axis];
                         });
        nodes_[id].left = build(begin, mid);
        nodes_[id].right = build(mid, end);
        return id;
    }

    Hit trace(const Ray& ray, bool any) const {
        Hit hit;
        if (nodes_.empty()) return hit;
        double distance = ray.t_max;
        const Point origin{ray.origin.x, ray.origin.y, ray.origin.z};
        const Point direction{ray.direction.x, ray.direction.y, ray.direction.z};
        const WideVec origin_wide = widen(ray.origin), direction_wide = widen(ray.direction);
        // Median splits bound the pending sibling stack by the index bit width.
        std::array<std::size_t, std::numeric_limits<std::size_t>::digits> stack{};
        std::size_t pending = 1;
        while (pending) {
            const auto& node = nodes_[stack[--pending]];
            const float bound = std::nextafter(static_cast<float>(distance), infinity);
            if (!node.bounds.intersects(origin, direction, ray.t_min, bound)) continue;
            if (node.left != 0) {
                stack[pending++] = node.left;
                stack[pending++] = node.right;
                continue;
            }
            for (auto i = node.begin; i < node.end; ++i) {
                const auto& p = primitives_[i];
                const WideVec h = cross(direction_wide, p.edge2);
                const double det = dot(p.edge1, h);
                if (det == 0) continue;
                const WideVec relative = origin_wide - p.vertex;
                const double u = dot(relative, h) / det;
                if (!(u >= 0 && u <= 1)) continue;
                const WideVec q = cross(relative, p.edge1);
                const double v = dot(direction_wide, q) / det;
                if (!(v >= 0 && u + v <= 1)) continue;
                const double t = dot(p.edge2, q) / det;
                if (!(t >= ray.t_min && t <= distance)) continue;
                hit = {p.surface, p.triangle, static_cast<float>(t), static_cast<float>(u),
                       static_cast<float>(v), p.normal, narrow(p.vertex + p.edge1 * u + p.edge2 * v)};
                if (any) return hit;
                distance = t;
            }
        }
        return hit;
    }

    std::vector<Primitive> primitives_;
    std::vector<Node> nodes_;
};

}

std::unique_ptr<Traversal> make_builtin(std::span<const SurfaceGeometry> surfaces) {
    return std::make_unique<Builtin>(surfaces);
}

}
