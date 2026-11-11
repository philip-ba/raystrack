#include "shortwave.hpp"
#include "vector.hpp"
#include "scattering.hpp"

#include <algorithm>
#include <limits>
#include <numbers>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace raystrack {
namespace {

constexpr double pi = std::numbers::pi;
constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr std::size_t batch_capacity = 65536;
constexpr int components = 4;

enum Component { sun, sky, reflected, transmitted };

bool positive(const RGB& value) { return value[0] > 0 || value[1] > 0 || value[2] > 0; }

void add(RGB& target, const RGB& source, double factor) {
    for (int k = 0; k < 3; ++k) target[k] += source[k] * factor;
}

RGB multiply(const RGB& a, const RGB& b) {
    return {a[0]*b[0], a[1]*b[1], a[2]*b[2]};
}

Vec3 offset(Vec3 point, Vec3 normal, float epsilon) {
    const auto away = [epsilon](float value, float direction) {
        if (direction == 0) return value;
        const float end = std::copysign(infinity, direction);
        const float shifted = std::nextafter(value + direction * epsilon, end);
        return std::isfinite(shifted) ? shifted : value;
    };
    return {away(point.x, normal.x), away(point.y, normal.y), away(point.z, normal.z)};
}

Ray ray(Vec3 origin, Vec3 direction) { return {origin, direction, 0, infinity}; }

struct Frame {
    Vec3 normal, u, v;

    explicit Frame(Vec3 n) : normal(n) {
        u = normalize(cross(std::abs(n.z) < 0.9f ? Vec3{0, 0, 1} : Vec3{0, 1, 0}, n));
        v = cross(n, u);
    }

    Vec3 cosine(float r1, float r2) const {
        const float radius = std::sqrt(r1), angle = static_cast<float>(2*pi) * r2;
        return normalize(u * (radius * std::cos(angle)) + v * (radius * std::sin(angle))
                         + normal * std::sqrt(1 - r1));
    }
};

class SkyLookup {
public:
    explicit SkyLookup(const Lighting& light) : light_(light) {}

    int index(Vec3 direction) const {
        if (direction.z < 0) return -1;
        if (!light_.rings) return 0;
        const auto row = std::min(light_.rings - 1, static_cast<std::size_t>(
            std::atan2(direction.z, std::hypot(direction.x, direction.y)) / light_.ring_height));
        const int start = light_.ring_offsets[row];
        const int count = light_.ring_offsets[row + 1] - start;
        const double azimuth = std::atan2(static_cast<double>(direction.x), direction.y) * (180 / pi)
                               + light_.north_degrees;
        const double width = 360.0 / count;
        const int column = static_cast<int>(std::floor((azimuth + width / 2) / width));
        return start + (column + count) % count;
    }

    const RGB& at(std::size_t step, int patch) const {
        return light_.sky[step * light_.sky_stride + patch];
    }

private:
    const Lighting& light_;
};

// Aggregate unrelated shadow rays into one backend dispatch, including across
// timesteps. The destination rows remain valid for the duration of the solve.
class ShadowBatch {
public:
    explicit ShadowBatch(const Traversal& traversal) : traversal_(traversal) {}

    void enqueue(Ray query, RGB& target, RGB value, double factor) {
        rays_.push_back(query);
        values_.push_back({&target, value, factor});
        if (rays_.size() == batch_capacity) flush();
    }

    void flush() {
        if (rays_.empty()) return;
        blocked_.resize(rays_.size());
        traversal_.occluded(rays_, blocked_);
        for (std::size_t i = 0; i < rays_.size(); ++i) {
            const auto& value = values_[i];
            if (!blocked_[i]) add(*value.target, value.color, value.factor);
        }
        rays_.clear();
        values_.clear();
    }

private:
    struct Contribution { RGB* target; RGB color; double factor; };
    const Traversal& traversal_;
    std::vector<Ray> rays_;
    std::vector<Contribution> values_;
    std::vector<std::uint8_t> blocked_;
};

// Delta sun paths are disjoint from finite-lobe next-event estimates. Image
// directions locate their zero-measure sensor directions; geometry traversal
// then verifies each candidate and accounts for every sheet interaction.
class DeltaConnections {
public:
    struct Direction {
        double x, y, z;
        explicit Direction(Vec3 value) : x(value.x), y(value.y), z(value.z) { normalize_self(); }
        Direction(double a, double b, double c) : x(a), y(b), z(c) { normalize_self(); }
        Vec3 vector() const { return {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)}; }
        Direction reflect(Vec3 value) const {
            const Direction normal(value);
            const double twice = 2*(x*normal.x + y*normal.y + z*normal.z);
            return Direction(x-twice*normal.x, y-twice*normal.y, z-twice*normal.z);
        }
        bool same(const Direction& other, double tolerance = 1e-12) const {
            return std::abs(x-other.x) <= tolerance && std::abs(y-other.y) <= tolerance
                && std::abs(z-other.z) <= tolerance;
        }
    private:
        void normalize_self() {
            const double length = std::hypot(x, y, z);
            x /= length; y /= length; z /= length;
        }
    };

    DeltaConnections(const Traversal& traversal, std::span<const OpticalMaterial> materials, float epsilon)
        : traversal_(traversal), materials_(materials), epsilon_(epsilon) {}

    const std::vector<Direction>& directions(Vec3 sun_direction, std::uint32_t remaining) {
        if (!normals_ready_ && remaining) {
            for (const auto& material : materials_) {
                if (material.kind != 1 || material.refractive_index == 1) continue;
                for (std::size_t i = 0; i < material.delta_normal_count; ++i) {
                    const auto normal = material.delta_normals[i];
                    if (std::none_of(normals_.begin(), normals_.end(), [&](Vec3 old) {
                        return (old.x == normal.x && old.y == normal.y && old.z == normal.z)
                            || (old.x == -normal.x && old.y == -normal.y && old.z == -normal.z);
                    })) {
                        if (normals_.size() == 4096)
                            throw std::runtime_error("capability: clear glazing exceeds 4096 distinct mirror normals");
                        normals_.push_back(normal);
                    }
                }
            }
            normals_ready_ = true;
        }
        const auto key = std::tuple{sun_direction.x, sun_direction.y, sun_direction.z, remaining};
        auto [entry, inserted] = candidates_.try_emplace(key);
        if (!inserted) return entry->second;
        auto& values = entry->second;
        values.emplace_back(sun_direction);
        std::size_t attempts = 0;
        std::size_t begin = 0, end = 1;
        for (std::uint32_t level = 0; level < remaining; ++level) {
            for (std::size_t i = begin; i < end; ++i) {
                for (const auto normal : normals_) {
                    if (++attempts > 1048576)
                        throw std::runtime_error("capability: clear glazing exceeds 1048576 image construction steps");
                    const auto candidate = values[i].reflect(normal);
                    if (std::any_of(values.begin(), values.end(), [&](const Direction& old) { return old.same(candidate); })) continue;
                    if (std::any_of(values.begin(), values.end(), [&](const Direction& old) { return old.same(candidate, 2e-6); }))
                        throw std::runtime_error("capability: clear glazing image directions are unresolved at float ray precision");
                    if (values.size() == 4096)
                        throw std::runtime_error("capability: clear glazing exceeds 4096 image directions");
                    values.push_back(candidate);
                }
            }
            begin = end;
            end = values.size();
            if (begin == end) break;
        }
        cached_directions_ += values.size();
        if (cached_directions_ > 1048576)
            throw std::runtime_error("capability: clear glazing exceeds 1048576 cached image directions");
        return values;
    }

    void enqueue(Ray query, Direction direction, Vec3 sun_direction, RGB* target, RGB value,
                 Component first, std::uint32_t remaining) {
        roots_.push_back({Direction(sun_direction), target, 0});
        pending_.push_back({query, direction, value, first, remaining, roots_.size() - 1});
        if (pending_.size() >= batch_capacity) flush();
    }

    void flush() {
        std::vector<Node> next;
        std::vector<Ray> rays;
        std::vector<Hit> hits;
        while (!pending_.empty()) {
            rays.clear();
            for (const auto& node : pending_) rays.push_back(node.query);
            hits.resize(rays.size());
            traversal_.intersect(rays, hits);
            next.clear();
            for (std::size_t i = 0; i < pending_.size(); ++i) {
                const auto& node = pending_[i];
                auto& root = roots_[node.root];
                if (++root.work > 65536)
                    throw std::runtime_error("capability: clear glazing exceeds 65536 path branches");
                const auto& hit = hits[i];
                if (hit.surface < 0) {
                    if (node.direction.same(root.sun))
                        add(root.target[node.first == sky ? sun : node.first], node.value, 1);
                    continue;
                }
                const auto& material = materials_[hit.surface];
                if (!node.remaining || material.kind != 1) continue;
                const double transmission = plate_transmission(dot(node.query.direction, hit.normal), material.refractive_index);
                const Vec3 facing = hit.normal * (dot(node.query.direction, hit.normal) < 0 ? 1 : -1);
                for (const bool reflection : {false, true}) {
                    RGB value = node.value;
                    for (auto& channel : value) channel *= reflection ? 1 - transmission : transmission;
                    if (!positive(value)) continue;
                    const Direction direction = reflection ? node.direction.reflect(canonical_normal(material, hit.normal)) : node.direction;
                    const Component first = node.first == sky ? (reflection ? reflected : transmitted) : node.first;
                    if (next.size() == 1048576)
                        throw std::runtime_error("capability: clear glazing exceeds 1048576 active connection branches");
                    next.push_back({ray(offset(hit.point, facing * (reflection ? 1 : -1), epsilon_), direction.vector()),
                                    direction, value, first, node.remaining - 1, node.root});
                }
            }
            pending_.swap(next);
        }
        roots_.clear();
    }

private:
    // Exact specular direction algebra uses doubles. Float rays only locate
    // geometry; unresolved distinct image directions fail explicitly above.
    struct Root { Direction sun; RGB* target; std::size_t work; };
    struct Node { Ray query; Direction direction; RGB value; Component first; std::uint32_t remaining; std::size_t root; };
    const Traversal& traversal_;
    std::span<const OpticalMaterial> materials_;
    float epsilon_;
    std::vector<Vec3> normals_;
    bool normals_ready_ = false;
    std::size_t cached_directions_ = 0;
    std::map<std::tuple<float, float, float, std::uint32_t>, std::vector<Direction>> candidates_;
    std::vector<Root> roots_;
    std::vector<Node> pending_;

    static Vec3 canonical_normal(const OpticalMaterial& material, Vec3 actual) {
        const Direction measured(actual);
        Vec3 matched{};
        bool found = false;
        for (std::size_t i = 0; i < material.delta_normal_count; ++i) {
            const Vec3 normal = material.delta_normals[i];
            if (!measured.same(Direction(normal), 2e-6) && !measured.same(Direction(normal * -1), 2e-6)) continue;
            if (found)
                throw std::runtime_error("capability: clear glazing hit normal is ambiguous at float ray precision");
            matched = normal;
            found = true;
        }
        if (found) return matched;
        throw std::runtime_error("capability: clear glazing hit normal does not match its image directions");
    }
};

float uniform(std::mt19937_64& rng) {
    return static_cast<float>(rng() >> 40) * 0x1p-24f;
}

double uniform_double(std::mt19937_64& rng) {
    return static_cast<double>(rng() >> 11) * 0x1p-53;
}

Vec3 sample(const Frame& frame, std::mt19937_64& rng) {
    const float r1 = uniform(rng), r2 = uniform(rng);
    return frame.cosine(r1, r2);
}

constexpr int klems_patches = 145;
constexpr std::array klems_bounds{0, 5, 15, 25, 35, 45, 55, 65, 75, 90};
constexpr std::array klems_counts{1, 8, 16, 20, 24, 24, 24, 16, 12};
constexpr std::array klems_start{0, 1, 9, 25, 45, 69, 93, 117, 133, 145};

int klems_patch(Vec3 local) {
    const double degrees = std::atan2(std::hypot(local.x, local.y), std::abs(local.z)) * (180 / pi);
    const auto row = std::clamp(std::upper_bound(klems_bounds.begin(), klems_bounds.end(), degrees)
                                - klems_bounds.begin() - 1, std::ptrdiff_t{0}, std::ptrdiff_t{8});
    const auto count = klems_counts[row];
    const double azimuth = std::atan2(static_cast<double>(local.y), local.x) * (180 / pi);
    const double width = 360.0 / count;
    const int column = static_cast<int>(std::floor((azimuth + width / 2) / width));
    return klems_start[row] + (column + count) % count;
}

Vec3 klems_direction(int patch, bool front, std::mt19937_64& rng) {
    const auto row = std::upper_bound(klems_start.begin(), klems_start.end(), patch) - klems_start.begin() - 1;
    const double low = std::cos(klems_bounds[row + 1] * pi / 180);
    const double high = std::cos(klems_bounds[row] * pi / 180);
    const double z_squared = low*low + uniform(rng) * (high*high - low*low);
    const double azimuth = (patch - klems_start[row] + static_cast<double>(uniform(rng)) - 0.5)
                           * (2*pi / klems_counts[row]);
    const double radius = std::sqrt(1 - z_squared);
    return {static_cast<float>(-radius * std::cos(azimuth)), static_cast<float>(-radius * std::sin(azimuth)),
            static_cast<float>((front ? 1 : -1) * std::sqrt(z_squared))};
}

struct Path {
    Ray query;
    RGB weight;
    std::size_t sensor;
    Component first;
    std::int32_t medium = -1;
    double skipped_distance = 0;
};

struct Event {
    Vec3 point, winding, normal, tangent{}, bitangent{};
    const OpticalMaterial& material;
    Path path;
    bool front;
    int outgoing = 0;
    std::int32_t solid_surface = -1;
    DielectricEvent solid{};

    Event(const Hit& hit, const Path& source, const OpticalMaterial& optics)
        : point(hit.point), winding(hit.normal), material(optics), path(source),
          front(dot(winding, source.query.direction) < 0) {
        normal = winding * (front ? 1 : -1);
        if (material.kind == 2)
            solid = dielectric(source.query.direction, normal, material.refractive_index, front);
        if (!material.klems) return;
        const auto projection = [this](Vec3 axis) { return axis - winding * dot(axis, winding); };
        Vec3 axis = projection(material.tangent);
        if (dot(axis, axis) < 1e-8f) axis = projection(material.alternate);
        tangent = normalize(axis);
        bitangent = cross(winding, tangent);
        outgoing = klems_patch(local(source.query.direction * -1));
    }

    Vec3 local(Vec3 direction) const {
        return {dot(direction, tangent), dot(direction, bitangent), dot(direction, winding)};
    }

    int lobe(bool reflection) const {
        const bool incident_front = reflection ? front : !front;
        return (reflection ? 0 : 2) + (incident_front ? 0 : 1);
    }

    const RGB& diffuse(bool reflection) const {
        if (!reflection) return material.transmission;
        return front ? material.front : material.back;
    }

    RGB evaluate(Vec3 direction, bool reflection) const {
        if (!material.klems) return diffuse(reflection);
        const Vec3 incident = local(direction);
        const int patch = klems_patch({-incident.x, -incident.y, incident.z});
        const double scaled = pi * material.klems[(lobe(reflection) * klems_patches + patch) * klems_patches + outgoing];
        return {scaled, scaled, scaled};
    }

    const double* cdf(bool reflection) const {
        return material.reverse_cdf + (lobe(reflection) * klems_patches + outgoing) * klems_patches;
    }

    RGB integrated(bool reflection) const {
        if (material.kind == 2) {
            const double value = reflection ? solid.reflectance : 1-solid.reflectance;
            return {value, value, value};
        }
        if (material.kind == 1) {
            const double transmission = plate_transmission(dot(path.query.direction, winding), material.refractive_index);
            const double value = reflection ? 1 - transmission : transmission;
            return {value, value, value};
        }
        if (!material.klems) return diffuse(reflection);
        const double integral = cdf(reflection)[klems_patches - 1];
        return {integral, integral, integral};
    }

    Vec3 direction(bool reflection, const RGB& integral, std::mt19937_64& rng) const {
        if (material.kind == 2)
            return reflection ? specular_reflection(path.query.direction, winding) : solid.transmission;
        if (material.kind == 1)
            return reflection ? specular_reflection(path.query.direction, winding) : path.query.direction;
        if (!material.klems) return sample(Frame(normal * (reflection ? 1 : -1)), rng);
        const double* first = cdf(reflection);
        const double target = std::min(uniform_double(rng) * integral[0], std::nextafter(integral[0], 0.0));
        const auto patch = std::upper_bound(first, first + klems_patches, target) - first;
        const Vec3 sampled = klems_direction(static_cast<int>(patch), reflection ? front : !front, rng);
        return normalize(tangent * sampled.x + bitangent * sampled.y + winding * sampled.z);
    }
};

double magnitude(const RGB& value) { return *std::max_element(value.begin(), value.end()); }

void continue_path(const Event& event, bool reflection, RGB weight, double probability,
                   float epsilon, std::mt19937_64& rng, std::vector<Path>& next) {
    if (!positive(weight)) return;
    const Vec3 direction = event.direction(reflection, weight, rng);
    const Vec3 side = event.normal * (reflection ? 1 : -1);
    for (int i = 0; i < 3; ++i) weight[i] *= event.path.weight[i] / probability;
    auto medium = event.path.medium;
    Vec3 origin = offset(event.point, side, epsilon);
    double skipped = 0;
    if (event.material.kind == 2) {
        if (!reflection) {
            for (auto& value : weight) value *= event.solid.eta*event.solid.eta;
            medium = event.front ? event.solid_surface : -1;
        }
        // A normal offset avoids float self-hits even at grazing incidence.
        // Restore the omitted in-medium distance from the displaced ray origin
        // to the boundary plane, measured along the new ray direction.
        const double projection = dot(direction, event.winding);
        if (projection != 0)
            skipped = dot(origin-event.point, event.winding) / projection;
    }
    next.push_back({ray(origin, direction), weight, event.path.sensor,
                    event.path.first == sky ? (reflection ? reflected : transmitted) : event.path.first,
                    medium, skipped});
}

}

void shortwave(const Traversal& traversal, std::span<const Sensor> sensors,
               const Lighting& light, std::size_t steps,
               std::span<const OpticalMaterial> materials, std::uint32_t samples,
               std::uint64_t seed, std::uint32_t bounces, float epsilon, RGB* result) {
    if (sensors.empty() || !steps) return;
    const auto row = [=](std::size_t step, std::size_t sensor) {
        return result + (step * sensors.size() + sensor) * components;
    };
    std::fill_n(result, steps * sensors.size() * components, RGB{});
    ShadowBatch shadows(traversal);
    const bool has_delta = std::any_of(materials.begin(), materials.end(),
                                       [](const auto& material) { return material.kind == 1; });
    DeltaConnections connections(traversal, materials, epsilon);
    std::vector<Vec3> sensor_origins;
    for (const auto& sensor : sensors) sensor_origins.push_back(offset(sensor.point, sensor.normal, epsilon));
    std::vector<std::size_t> sun_steps;
    for (std::size_t step = 0; step < steps; ++step) {
        const Vec3 direction = light.sun[step];
        if (direction.z <= 0 || !positive(light.dni[step])) continue;
        sun_steps.push_back(step);
        for (std::size_t i = 0; i < sensors.size(); ++i) {
            const auto& sensor = sensors[i];
            const float cosine = dot(sensor.normal, direction);
            if (!has_delta) {
                if (cosine > 0) shadows.enqueue(ray(sensor_origins[i], direction),
                                                row(step, i)[sun], light.dni[step], cosine);
            } else {
                for (const auto candidate : connections.directions(direction, bounces)) {
                    const float projection = dot(sensor.normal, candidate.vector());
                    if (projection <= 0) continue;
                    RGB value = light.dni[step];
                    for (auto& channel : value) channel *= projection;
                    connections.enqueue(ray(sensor_origins[i], candidate.vector()), candidate, direction,
                                        row(step, i), value, sky, bounces);
                }
            }
        }
    }
    shadows.flush();
    connections.flush();
    if (!samples) return;

    const auto patch_count = light.rings ? light.ring_offsets[light.rings] : 1;
    std::vector<std::size_t> sky_steps;
    for (std::size_t step = 0; step < steps; ++step) {
        const RGB* first = light.sky + step * light.sky_stride;
        if (std::any_of(first, first + patch_count, positive)) sky_steps.push_back(step);
    }
    if (sky_steps.empty() && (!bounces || sun_steps.empty())) return;

    SkyLookup sky_map(light);
    std::mt19937_64 rng(seed);
    std::vector<Path> paths, next;
    std::vector<Ray> queries;
    std::vector<Hit> hits;
    std::vector<Event> events;
    const double weight = 1.0 / samples;

    const auto process = [&] {
        // Reduce sky transfer by sensor and patch before applying the timestep
        // radiance series. The ordered map fixes the reduction order.
        std::map<std::tuple<std::size_t, int, Component>, RGB> transfer;
        for (std::uint32_t depth = 0; depth <= bounces && !paths.empty(); ++depth) {
            queries.clear();
            for (const auto& path : paths) queries.push_back(path.query);
            hits.resize(paths.size());
            traversal.intersect(queries, hits);
            events.clear();
            next.clear();
            for (std::size_t i = 0; i < paths.size(); ++i) {
                const auto& hit = hits[i];
                auto path = paths[i];
                if (hit.surface < 0) {
                    if (path.medium >= 0)
                        throw std::runtime_error("capability: SolidDielectric ray escaped without an exit boundary");
                    const int patch = sky_map.index(path.query.direction);
                    if (patch >= 0 && !sky_steps.empty())
                        add(transfer[{path.sensor, patch, path.first}], path.weight, pi);
                    continue;
                }
                const auto& material = materials[hit.surface];
                if (path.medium >= 0) {
                    if (hit.surface != path.medium || material.kind != 2)
                        throw std::runtime_error("capability: SolidDielectric nesting, overlap or interior objects are unsupported");
                    const double distance = hit.distance + path.skipped_distance;
                    for (int channel = 0; channel < 3; ++channel)
                        path.weight[channel] *= std::exp(-material.absorption[channel]*distance);
                }
                if (material.kind == 2) {
                    const bool entering = dot(path.query.direction, hit.normal) < 0;
                    if ((entering && path.medium >= 0) || (!entering && path.medium != hit.surface))
                        throw std::runtime_error("capability: SolidDielectric boundary winding or current medium is inconsistent: depth="
                            + std::to_string(depth) + " medium=" + std::to_string(path.medium)
                            + " primitive=" + std::to_string(hit.primitive) + " distance=" + std::to_string(hit.distance)
                            + " cosine=" + std::to_string(dot(path.query.direction, hit.normal))
                            + " z=" + std::to_string(hit.point.z) + " origin_z=" + std::to_string(path.query.origin.z)
                            + " dx=" + std::to_string(path.query.direction.x)
                            + " dy=" + std::to_string(path.query.direction.y)
                            + " dz=" + std::to_string(path.query.direction.z));
                }
                if (depth == bounces) continue;
                Event event(hit, path, material);
                if (material.kind == 2) event.solid_surface = hit.surface;
                const RGB rho = event.integrated(true), tau = event.integrated(false);
                if (!positive(rho) && !positive(tau)) continue;
                if (material.kind == 0) events.push_back(event);
                if (depth + 1 == bounces && sky_steps.empty()) continue;
                if (depth == 0) {
                    continue_path(event, true, rho, 1, epsilon, rng, next);
                    continue_path(event, false, tau, 1, epsilon, rng, next);
                } else {
                    const double probability = magnitude(rho) / (magnitude(rho) + magnitude(tau));
                    const bool reflection = uniform_double(rng) < probability;
                    continue_path(event, reflection, reflection ? rho : tau,
                                  reflection ? probability : 1 - probability, epsilon, rng, next);
                }
            }

            // Connect the narrow direct sun explicitly at each scatter. The
            // first interaction determines the output component at the sensor.
            for (auto step : sun_steps) {
                const Vec3 direction = light.sun[step];
                for (const auto& event : events) {
                    if (has_delta) {
                        const auto remaining = bounces - depth - 1;
                        for (const auto candidate : connections.directions(direction, remaining)) {
                            const float cosine = dot(candidate.vector(), event.normal);
                            if (cosine == 0) continue;
                            const bool reflection = cosine > 0;
                            const RGB lobe = event.evaluate(candidate.vector(), reflection);
                            if (!positive(lobe)) continue;
                            const auto& path = event.path;
                            const Component component = path.first == sky ? (reflection ? reflected : transmitted) : path.first;
                            RGB value = multiply(multiply(light.dni[step], lobe), path.weight);
                            for (auto& channel : value) channel *= std::abs(cosine);
                            const Vec3 side = event.normal * (reflection ? 1 : -1);
                            connections.enqueue(ray(offset(event.point, side, epsilon), candidate.vector()), candidate, direction,
                                                row(step, path.sensor), value, component, remaining);
                        }
                        continue;
                    }
                    const float cosine = dot(direction, event.normal);
                    if (cosine == 0) continue;
                    const bool reflection = cosine > 0;
                    const RGB lobe = event.evaluate(direction, reflection);
                    if (!positive(lobe)) continue;
                    const auto& path = event.path;
                    const Component component = path.first == sky ? (reflection ? reflected : transmitted) : path.first;
                    const Vec3 side = event.normal * (reflection ? 1 : -1);
                    shadows.enqueue(ray(offset(event.point, side, epsilon), direction), row(step, path.sensor)[component],
                                    multiply(multiply(light.dni[step], lobe), path.weight), std::abs(cosine));
                }
            }
            paths.swap(next);
        }
        for (const auto& [key, factor] : transfer) {
            const auto [sensor, patch, component] = key;
            for (auto step : sky_steps)
                add(row(step, sensor)[component], multiply(sky_map.at(step, patch), factor), 1);
        }

        paths.clear();
    };

    // An independent stream for primary directions keeps the sample pattern
    // stable across changes in the number of material interactions.
    std::mt19937_64 primary_rng(seed ^ 0xe7037ed1a0b428dbULL);
    for (std::size_t i = 0; i < sensors.size(); ++i) {
        const auto& sensor = sensors[i];
        const auto origin = sensor_origins[i];
        const Frame frame(sensor.normal);
        for (std::uint32_t j = 0; j < samples; ++j) {
            paths.push_back({ray(origin, sample(frame, primary_rng)), {weight, weight, weight}, i, sky});
            if (paths.size() == batch_capacity) process();
        }
    }
    if (!paths.empty()) process();
    shadows.flush();
    connections.flush();
}

}
