#include "raystrack/traversal.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace raystrack {
namespace {

constexpr uint32_t workgroup_size = 64;
using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Profile {
    bool enabled = std::getenv("RAYSTRACK_VULKAN_PROFILE") != nullptr;
    double context_ms{}, pipeline_ms{}, geometry_pack_ms{}, geometry_build_ms{};
    double upload_pack_ms{}, dispatch_wait_ms{}, readback_decode_ms{};
    uint64_t updates{}, dispatches{}, rays{};
};
constexpr uint32_t shader_code[] =
#include "trace_spirv.inc"
;

constexpr std::array required_extensions{
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
};

void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string("Vulkan ") + operation + " failed (VkResult "
                                 + std::to_string(result) + ")");
    }
}

struct RayFeatures {
    VkPhysicalDeviceRayQueryFeaturesKHR query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, &query};
    VkPhysicalDeviceBufferDeviceAddressFeatures address{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES, &acceleration};
};

bool supports_ray_queries(VkPhysicalDevice physical) {
    uint32_t count = 0;
    check(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr), "enumerate extensions");
    std::vector<VkExtensionProperties> extensions(count);
    check(vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data()),
          "enumerate extensions");
    for (const auto* name : required_extensions) {
        if (std::none_of(extensions.begin(), extensions.end(), [name](const auto& extension) {
                return std::string_view(extension.extensionName) == name;
            })) {
            return false;
        }
    }
    RayFeatures capabilities;
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &capabilities.address};
    vkGetPhysicalDeviceFeatures2(physical, &features);
    return capabilities.address.bufferDeviceAddress && capabilities.acceleration.accelerationStructure
        && capabilities.query.rayQuery;
}

struct Context {
    VkInstance instance{};
    VkDevice device{};
    VkQueue queue{};
    VkCommandPool pool{};
    VkCommandBuffer command{};
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memory{};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR acceleration_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    PFN_vkCreateAccelerationStructureKHR create_acceleration{};
    PFN_vkDestroyAccelerationStructureKHR destroy_acceleration{};
    PFN_vkGetAccelerationStructureBuildSizesKHR build_sizes{};
    PFN_vkGetAccelerationStructureDeviceAddressKHR acceleration_address{};
    PFN_vkCmdBuildAccelerationStructuresKHR build_acceleration{};

    ~Context() {
        if (device) {
            vkDestroyCommandPool(device, pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) {
            vkDestroyInstance(instance, nullptr);
        }
    }

    void initialize() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "Raystrack";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instance_info.pApplicationInfo = &app;
        check(vkCreateInstance(&instance_info, nullptr, &instance), "create instance");

        uint32_t count = 0;
        check(vkEnumeratePhysicalDevices(instance, &count, nullptr), "enumerate physical devices");
        std::vector<VkPhysicalDevice> devices(count);
        check(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "enumerate physical devices");

        VkPhysicalDevice physical{};
        uint32_t queue_family = 0;
        for (auto candidate : devices) {
            VkPhysicalDeviceProperties candidate_properties{};
            vkGetPhysicalDeviceProperties(candidate, &candidate_properties);
            const auto type = candidate_properties.deviceType;
            if (candidate_properties.apiVersion < VK_API_VERSION_1_2
                || (type != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && type != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                || !supports_ray_queries(candidate)) {
                continue;
            }
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, nullptr);
            std::vector<VkQueueFamilyProperties> families(count);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &count, families.data());
            const auto compute = std::find_if(families.begin(), families.end(), [](const auto& family) {
                return family.queueCount > 0 && (family.queueFlags & VK_QUEUE_COMPUTE_BIT);
            });
            if (compute != families.end()) {
                physical = candidate;
                queue_family = static_cast<uint32_t>(compute - families.begin());
                properties = candidate_properties;
                break;
            }
        }
        if (!physical) {
            throw std::runtime_error("Vulkan RT requires a physical GPU with Vulkan 1.2, compute, "
                                     "rayQuery, accelerationStructure and bufferDeviceAddress");
        }

        VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                                 &acceleration_properties};
        vkGetPhysicalDeviceProperties2(physical, &properties2);
        vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        constexpr float priority = 1;
        VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue_info.queueFamilyIndex = queue_family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &priority;
        RayFeatures features;
        features.address.bufferDeviceAddress = VK_TRUE;
        features.acceleration.accelerationStructure = VK_TRUE;
        features.query.rayQuery = VK_TRUE;
        VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &features.address};
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;
        device_info.enabledExtensionCount = static_cast<uint32_t>(required_extensions.size());
        device_info.ppEnabledExtensionNames = required_extensions.data();
        check(vkCreateDevice(physical, &device_info, nullptr, &device), "create device");
        vkGetDeviceQueue(device, queue_family, 0, &queue);

        create_acceleration = load<PFN_vkCreateAccelerationStructureKHR>("vkCreateAccelerationStructureKHR");
        destroy_acceleration = load<PFN_vkDestroyAccelerationStructureKHR>("vkDestroyAccelerationStructureKHR");
        build_sizes = load<PFN_vkGetAccelerationStructureBuildSizesKHR>("vkGetAccelerationStructureBuildSizesKHR");
        acceleration_address = load<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
            "vkGetAccelerationStructureDeviceAddressKHR");
        build_acceleration = load<PFN_vkCmdBuildAccelerationStructuresKHR>("vkCmdBuildAccelerationStructuresKHR");

        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool_info.queueFamilyIndex = queue_family;
        check(vkCreateCommandPool(device, &pool_info, nullptr, &pool), "create command pool");
        VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_info.commandPool = pool;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device, &command_info, &command), "allocate command buffer");
    }

    uint32_t memory_type(uint32_t compatible, VkMemoryPropertyFlags flags) const {
        for (uint32_t index = 0; index < memory.memoryTypeCount; ++index) {
            if ((compatible & (1u << index)) && (memory.memoryTypes[index].propertyFlags & flags) == flags) {
                return index;
            }
        }
        throw std::runtime_error("Vulkan RT could not find compatible memory for a buffer");
    }

    VkCommandBuffer begin() const {
        check(vkResetCommandPool(device, pool, 0), "reset command pool");
        VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(command, &info), "begin command buffer");
        return command;
    }

    void submit() const {
        check(vkEndCommandBuffer(command), "end command buffer");
        VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        info.commandBufferCount = 1;
        info.pCommandBuffers = &command;
        check(vkQueueSubmit(queue, 1, &info, VK_NULL_HANDLE), "submit work");
        check(vkQueueWaitIdle(queue), "wait for work");
    }

private:
    template <class Function>
    Function load(const char* name) const {
        const auto pointer = vkGetDeviceProcAddr(device, name);
        if (!pointer) {
            throw std::runtime_error(std::string("Vulkan RT missing device function ") + name);
        }
        return reinterpret_cast<Function>(pointer);
    }
};

constexpr auto host_memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr auto address_usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
constexpr auto build_input = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | address_usage;

struct Buffer {
    const Context& context;
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    void* data{};

    explicit Buffer(const Context& owner) : context(owner) {}
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    ~Buffer() {
        if (data) {
            vkUnmapMemory(context.device, memory);
        }
        if (buffer) {
            vkDestroyBuffer(context.device, buffer, nullptr);
        }
        if (memory) {
            vkFreeMemory(context.device, memory, nullptr);
        }
    }

    void create(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(context.device, &info, nullptr, &buffer), "create buffer");
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(context.device, buffer, &requirements);
        VkMemoryAllocateFlagsInfo address_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        address_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo memory_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        memory_info.pNext = usage & address_usage ? &address_info : nullptr;
        memory_info.allocationSize = requirements.size;
        memory_info.memoryTypeIndex = context.memory_type(requirements.memoryTypeBits, flags);
        check(vkAllocateMemory(context.device, &memory_info, nullptr, &memory), "allocate buffer memory");
        check(vkBindBufferMemory(context.device, buffer, memory, 0), "bind buffer memory");
        if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            check(vkMapMemory(context.device, memory, 0, size, 0, &data), "map buffer memory");
        }
    }

    VkDeviceAddress address() const {
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = buffer;
        return vkGetBufferDeviceAddress(context.device, &info);
    }
};

struct AccelerationStructure {
    Buffer storage;
    VkAccelerationStructureKHR handle{};

    explicit AccelerationStructure(const Context& owner) : storage(owner) {}
    ~AccelerationStructure() {
        if (handle) {
            storage.context.destroy_acceleration(storage.context.device, handle, nullptr);
        }
    }

    VkAccelerationStructureBuildGeometryInfoKHR prepare(VkAccelerationStructureTypeKHR type,
        const VkAccelerationStructureGeometryKHR& geometry, uint32_t count, Buffer& scratch) {
        const auto& context = storage.context;
        VkAccelerationStructureBuildGeometryInfoKHR build{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        build.type = type;
        build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build.geometryCount = 1;
        build.pGeometries = &geometry;
        VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        context.build_sizes(context.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build, &count, &sizes);

        storage.create(sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                       | address_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkAccelerationStructureCreateInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        info.type = type;
        info.buffer = storage.buffer;
        info.size = sizes.accelerationStructureSize;
        check(context.create_acceleration(context.device, &info, nullptr, &handle), "create acceleration structure");

        const auto alignment = context.acceleration_properties.minAccelerationStructureScratchOffsetAlignment;
        scratch.create(sizes.buildScratchSize + alignment, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | address_usage,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        build.dstAccelerationStructure = handle;
        build.scratchData.deviceAddress = ((scratch.address() + alignment - 1) / alignment) * alignment;
        return build;
    }

    VkDeviceAddress address() const {
        VkAccelerationStructureDeviceAddressInfoKHR info{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
        info.accelerationStructure = handle;
        return storage.context.acceleration_address(storage.context.device, &info);
    }
};

struct Primitive {
    int32_t surface;
    int32_t index;
    Vec3 normal;
};

Vec3 normal_for(const Vec3& a, const Vec3& b, const Vec3& c) {
    const double ux = static_cast<double>(b.x) - a.x;
    const double uy = static_cast<double>(b.y) - a.y;
    const double uz = static_cast<double>(b.z) - a.z;
    const double vx = static_cast<double>(c.x) - a.x;
    const double vy = static_cast<double>(c.y) - a.y;
    const double vz = static_cast<double>(c.z) - a.z;
    const double x = uy * vz - uz * vy;
    const double y = uz * vx - ux * vz;
    const double z = ux * vy - uy * vx;
    const double length = std::hypot(x, y, z);
    if (length == 0) {
        return {};
    }
    return {static_cast<float>(x / length), static_cast<float>(y / length), static_cast<float>(z / length)};
}

Vec3 hit_point(const Vec3& a, const Vec3& b, const Vec3& c, double u, double v) {
    // Reconstruct on the committed triangle. The hardware's hit distance can
    // lose enough precision to put a secondary ray on the wrong side.
    const auto coordinate = [u, v](double origin, double first, double second) {
        return static_cast<float>(origin + (first - origin) * u + (second - origin) * v);
    };
    return {coordinate(a.x, b.x, c.x), coordinate(a.y, b.y, c.y), coordinate(a.z, b.z, c.z)};
}

void barrier(VkCommandBuffer command, VkPipelineStageFlags source_stage, VkPipelineStageFlags target_stage,
             VkAccessFlags source_access, VkAccessFlags target_access) {
    VkMemoryBarrier memory_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memory_barrier.srcAccessMask = source_access;
    memory_barrier.dstAccessMask = target_access;
    vkCmdPipelineBarrier(command, source_stage, target_stage, 0, 1, &memory_barrier, 0, nullptr, 0, nullptr);
}

struct DeviceRay {
    std::array<float, 4> origin_and_min;
    std::array<float, 4> direction_and_max;
};

struct DeviceHit {
    int32_t primitive{-1};
    float distance{};
    float u{};
    float v{};
};

static_assert(sizeof(DeviceRay) == 32 && sizeof(DeviceHit) == 16);
static_assert(sizeof(Vec3) == 3 * sizeof(float) && sizeof(Triangle) == 3 * sizeof(uint32_t));

struct BatchBuffers {
    size_t capacity;
    Buffer input;
    Buffer output;

    BatchBuffers(const Context& context, size_t count) : capacity(count), input(context), output(context) {
        input.create(count * sizeof(DeviceRay), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_memory);
        output.create(count * sizeof(DeviceHit), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_memory);
    }
};

class VulkanTraversal final : public Traversal {
public:
    ~VulkanTraversal() override {
        if (profile_.enabled) {
            std::cerr << "{\"raystrack_vulkan_profile\":{\"device\":\"" << context_.properties.deviceName
                      << "\",\"api_version\":" << context_.properties.apiVersion
                      << ",\"driver_version\":" << context_.properties.driverVersion
                      << ",\"vendor_id\":" << context_.properties.vendorID
                      << ",\"device_id\":" << context_.properties.deviceID
                      << ",\"context_ms\":" << profile_.context_ms
                      << ",\"pipeline_ms\":" << profile_.pipeline_ms
                      << ",\"geometry_pack_ms\":" << profile_.geometry_pack_ms
                      << ",\"geometry_build_ms\":" << profile_.geometry_build_ms
                      << ",\"upload_pack_ms\":" << profile_.upload_pack_ms
                      << ",\"dispatch_wait_ms\":" << profile_.dispatch_wait_ms
                      << ",\"readback_decode_ms\":" << profile_.readback_decode_ms
                      << ",\"updates\":" << profile_.updates
                      << ",\"dispatches\":" << profile_.dispatches
                      << ",\"rays\":" << profile_.rays << "}}\n";
        }
        destroy_pipeline();
    }

    void initialize(std::span<const SurfaceGeometry> surfaces) {
        const auto start = Clock::now();
        context_.initialize();
        profile_.context_ms += elapsed_ms(start);
        update(surfaces);
    }

    bool update(std::span<const SurfaceGeometry> surfaces) override {
        const std::scoped_lock lock(mutex_);
        const auto pack_start = Clock::now();
        std::vector<Vec3> vertices;
        std::vector<Triangle> triangles;
        std::vector<Primitive> primitives;
        for (size_t surface = 0; surface < surfaces.size(); ++surface) {
            const auto& geometry = surfaces[surface];
            if (surface > static_cast<size_t>(std::numeric_limits<int32_t>::max())
                || geometry.vertices.size() > std::numeric_limits<uint32_t>::max() - vertices.size()
                || geometry.triangles.size() > std::numeric_limits<int32_t>::max() - triangles.size()
                || triangles.size() + geometry.triangles.size() > context_.acceleration_properties.maxPrimitiveCount) {
                throw std::invalid_argument("Scene exceeds Vulkan RT triangle or vertex limits");
            }
            const auto offset = static_cast<uint32_t>(vertices.size());
            for (const auto& vertex : geometry.vertices) {
                if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y) || !std::isfinite(vertex.z)) {
                    throw std::invalid_argument("Vulkan RT requires finite vertices");
                }
                vertices.push_back(vertex);
            }
            for (size_t index = 0; index < geometry.triangles.size(); ++index) {
                const auto triangle = geometry.triangles[index];
                if (std::max({triangle.a, triangle.b, triangle.c}) >= geometry.vertices.size()) {
                    throw std::invalid_argument("Vulkan RT triangle index outside vertex range");
                }
                triangles.push_back({offset + triangle.a, offset + triangle.b, offset + triangle.c});
                primitives.push_back({static_cast<int32_t>(surface), static_cast<int32_t>(index),
                    normal_for(geometry.vertices[triangle.a], geometry.vertices[triangle.b],
                               geometry.vertices[triangle.c])});
            }
        }
        profile_.geometry_pack_ms += elapsed_ms(pack_start);
        auto bottom = std::make_unique<AccelerationStructure>(context_);
        auto top = std::make_unique<AccelerationStructure>(context_);
        if (!triangles.empty()) {
            const auto build_start = Clock::now();
            build(vertices, triangles, *bottom, *top);
            profile_.geometry_build_ms += elapsed_ms(build_start);
            if (!pipeline_) {
                const auto pipeline_start = Clock::now();
                create_pipeline();
                profile_.pipeline_ms += elapsed_ms(pipeline_start);
            }
        }
        // Build and validate off to the side. Failed edits leave the previous
        // acceleration structure and its host metadata consistent and usable.
        bottom_.swap(bottom);
        top_.swap(top);
        vertices_.swap(vertices);
        triangles_.swap(triangles);
        primitives_.swap(primitives);
        if (buffers_ && !primitives_.empty()) bind(buffers_->input, buffers_->output);
        ++profile_.updates;
        return true;
    }

    void intersect(std::span<const Ray> rays, std::span<Hit> hits) const override {
        if (rays.size() != hits.size()) {
            throw std::invalid_argument("Vulkan RT ray and hit batch sizes differ");
        }
        trace<false>(rays, [&](size_t index, const DeviceHit& result) {
            Hit hit{};
            if (result.primitive >= 0) {
                const auto slot = static_cast<size_t>(result.primitive);
                const auto& primitive = primitives_.at(slot);
                const auto& triangle = triangles_[slot];
                const auto point = hit_point(vertices_[triangle.a], vertices_[triangle.b], vertices_[triangle.c],
                                             result.u, result.v);
                hit = {primitive.surface, primitive.index, result.distance, result.u, result.v, primitive.normal, point};
            }
            hits[index] = hit;
        });
    }

    void occluded(std::span<const Ray> rays, std::span<uint8_t> hits) const override {
        if (rays.size() != hits.size()) {
            throw std::invalid_argument("Vulkan RT ray and occlusion batch sizes differ");
        }
        trace<true>(rays, [&](size_t index, uint32_t result) { hits[index] = static_cast<uint8_t>(result); });
    }

private:
    Context context_;
    std::unique_ptr<AccelerationStructure> bottom_;
    std::unique_ptr<AccelerationStructure> top_;
    VkShaderModule shader_{};
    VkDescriptorSetLayout layout_{};
    VkPipelineLayout pipeline_layout_{};
    VkDescriptorPool descriptors_{};
    VkDescriptorSet set_{};
    VkPipeline pipeline_{};
    std::vector<Vec3> vertices_;
    std::vector<Triangle> triangles_;
    std::vector<Primitive> primitives_;
    mutable std::mutex mutex_;
    mutable std::unique_ptr<BatchBuffers> buffers_;
    mutable Profile profile_;

    void build(std::span<const Vec3> vertices, std::span<const Triangle> triangles,
               AccelerationStructure& bottom, AccelerationStructure& top) {
        Buffer vertices_buffer(context_);
        vertices_buffer.create(vertices.size_bytes(), build_input, host_memory);
        std::memcpy(vertices_buffer.data, vertices.data(), vertices.size_bytes());
        Buffer index_buffer(context_);
        index_buffer.create(triangles.size_bytes(), build_input, host_memory);
        std::memcpy(index_buffer.data, triangles.data(), triangles.size_bytes());

        VkAccelerationStructureGeometryKHR bottom_geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        bottom_geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        bottom_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        auto& mesh = bottom_geometry.geometry.triangles;
        mesh.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        mesh.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        mesh.vertexData.deviceAddress = vertices_buffer.address();
        mesh.vertexStride = sizeof(Vec3);
        mesh.maxVertex = static_cast<uint32_t>(vertices.size() - 1);
        mesh.indexType = VK_INDEX_TYPE_UINT32;
        mesh.indexData.deviceAddress = index_buffer.address();

        const auto count = static_cast<uint32_t>(triangles.size());
        Buffer bottom_scratch(context_);
        const auto bottom_build = bottom.prepare(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                                                  bottom_geometry, count, bottom_scratch);
        VkAccelerationStructureInstanceKHR instance{};
        instance.transform.matrix[0][0] = 1;
        instance.transform.matrix[1][1] = 1;
        instance.transform.matrix[2][2] = 1;
        instance.mask = 0xff;
        instance.accelerationStructureReference = bottom.address();
        Buffer instance_buffer(context_);
        instance_buffer.create(sizeof(instance), build_input, host_memory);
        std::memcpy(instance_buffer.data, &instance, sizeof(instance));

        VkAccelerationStructureGeometryKHR top_geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        top_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        auto& instances = top_geometry.geometry.instances;
        instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        instances.data.deviceAddress = instance_buffer.address();
        Buffer top_scratch(context_);
        const auto top_build = top.prepare(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, top_geometry, 1, top_scratch);

        const auto command = context_.begin();
        const VkAccelerationStructureBuildRangeInfoKHR bottom_range{count, 0, 0, 0};
        const auto* bottom_ranges = &bottom_range;
        context_.build_acceleration(command, 1, &bottom_build, &bottom_ranges);
        barrier(command, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        const VkAccelerationStructureBuildRangeInfoKHR top_range{1, 0, 0, 0};
        const auto* top_ranges = &top_range;
        context_.build_acceleration(command, 1, &top_build, &top_ranges);
        barrier(command, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        context_.submit();
    }

    void destroy_pipeline() noexcept {
        if (context_.device) {
            vkDestroyPipeline(context_.device, pipeline_, nullptr);
            vkDestroyShaderModule(context_.device, shader_, nullptr);
            vkDestroyDescriptorPool(context_.device, descriptors_, nullptr);
            vkDestroyPipelineLayout(context_.device, pipeline_layout_, nullptr);
            vkDestroyDescriptorSetLayout(context_.device, layout_, nullptr);
        }
        pipeline_ = VK_NULL_HANDLE;
        shader_ = VK_NULL_HANDLE;
        descriptors_ = VK_NULL_HANDLE;
        pipeline_layout_ = VK_NULL_HANDLE;
        layout_ = VK_NULL_HANDLE;
        set_ = VK_NULL_HANDLE;
    }

    void create_pipeline() {
        // Empty scenes create their pipeline lazily on the first populated
        // commit. Release partial setup on failure so that commit can be retried.
        try {
        constexpr auto stage = VK_SHADER_STAGE_COMPUTE_BIT;
        const std::array bindings{
            VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, stage, nullptr},
            VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stage, nullptr},
            VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, stage, nullptr},
        };
        VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout.bindingCount = static_cast<uint32_t>(bindings.size());
        layout.pBindings = bindings.data();
        check(vkCreateDescriptorSetLayout(context_.device, &layout, nullptr, &layout_), "create descriptor layout");
        const VkPushConstantRange constants{stage, 0, 2 * sizeof(uint32_t)};
        VkPipelineLayoutCreateInfo pipeline_layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeline_layout.setLayoutCount = 1;
        pipeline_layout.pSetLayouts = &layout_;
        pipeline_layout.pushConstantRangeCount = 1;
        pipeline_layout.pPushConstantRanges = &constants;
        check(vkCreatePipelineLayout(context_.device, &pipeline_layout, nullptr, &pipeline_layout_),
              "create pipeline layout");

        const std::array pool_sizes{
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
        };
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = 1;
        pool.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
        pool.pPoolSizes = pool_sizes.data();
        check(vkCreateDescriptorPool(context_.device, &pool, nullptr, &descriptors_), "create descriptor pool");
        VkDescriptorSetAllocateInfo set{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set.descriptorPool = descriptors_;
        set.descriptorSetCount = 1;
        set.pSetLayouts = &layout_;
        check(vkAllocateDescriptorSets(context_.device, &set, &set_), "allocate descriptor set");

        VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader.codeSize = sizeof(shader_code);
        shader.pCode = shader_code;
        check(vkCreateShaderModule(context_.device, &shader, nullptr, &shader_), "create ray-query shader");
        VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, stage, shader_, "main"};
        pipeline.layout = pipeline_layout_;
        check(vkCreateComputePipelines(context_.device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &pipeline_),
              "create ray-query pipeline");
        } catch (...) {
            destroy_pipeline();
            throw;
        }
    }

    void bind(const Buffer& input, const Buffer& output) const {
        VkWriteDescriptorSetAccelerationStructureKHR acceleration{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
        acceleration.accelerationStructureCount = 1;
        acceleration.pAccelerationStructures = &top_->handle;
        const VkDescriptorBufferInfo input_info{input.buffer, 0, VK_WHOLE_SIZE};
        const VkDescriptorBufferInfo output_info{output.buffer, 0, VK_WHOLE_SIZE};
        const std::array writes{
            VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &acceleration, set_, 0, 0, 1,
                                 VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR},
            VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set_, 1, 0, 1,
                                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &input_info},
            VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set_, 2, 0, 1,
                                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &output_info},
        };
        vkUpdateDescriptorSets(context_.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    template <bool AnyHit, class StoreResult>
    void trace(std::span<const Ray> rays, StoreResult store_result) const {
        using Result = std::conditional_t<AnyHit, uint32_t, DeviceHit>;
        const std::scoped_lock lock(mutex_);
        if (primitives_.empty()) {
            for (size_t index = 0; index < rays.size(); ++index) {
                store_result(index, Result{});
            }
            return;
        }
        if (rays.empty()) {
            return;
        }
        const auto& limits = context_.properties.limits;
        const size_t limit = std::min({size_t{262144},
            static_cast<size_t>(limits.maxComputeWorkGroupCount[0]) * workgroup_size,
            limits.maxStorageBufferRange / sizeof(DeviceRay)});
        const size_t requested = std::min(rays.size(), limit);
        if (!buffers_ || buffers_->capacity < requested) {
            auto replacement = std::make_unique<BatchBuffers>(context_, std::min(std::bit_ceil(requested), limit));
            bind(replacement->input, replacement->output);
            buffers_ = std::move(replacement);
        }
        const auto capacity = buffers_->capacity;
        auto* input_bytes = static_cast<unsigned char*>(buffers_->input.data);
        const auto* output_bytes = static_cast<const unsigned char*>(buffers_->output.data);

        for (size_t begin = 0; begin < rays.size(); begin += capacity) {
            const auto count = static_cast<uint32_t>(std::min(capacity, rays.size() - begin));
            const auto upload_start = Clock::now();
            for (uint32_t index = 0; index < count; ++index) {
                const auto& [origin, direction, t_min, t_max] = rays[begin + index];
                const DeviceRay ray{{origin.x, origin.y, origin.z, t_min},
                    {direction.x, direction.y, direction.z, std::min(t_max, std::numeric_limits<float>::max())}};
                std::memcpy(input_bytes + index * sizeof(DeviceRay), &ray, sizeof(ray));
            }
            profile_.upload_pack_ms += elapsed_ms(upload_start);
            const auto dispatch_start = Clock::now();
            const auto command = context_.begin();
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
            vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &set_, 0, nullptr);
            const std::array constants{count, static_cast<uint32_t>(AnyHit)};
            vkCmdPushConstants(command, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants),
                               constants.data());
            vkCmdDispatch(command, (count + workgroup_size - 1) / workgroup_size, 1, 1);
            barrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            context_.submit();
            profile_.dispatch_wait_ms += elapsed_ms(dispatch_start);

            const auto readback_start = Clock::now();
            for (uint32_t index = 0; index < count; ++index) {
                Result result{};
                std::memcpy(&result, output_bytes + index * sizeof(Result), sizeof(result));
                store_result(begin + index, result);
            }
            profile_.readback_decode_ms += elapsed_ms(readback_start);
            ++profile_.dispatches;
            profile_.rays += count;
        }
    }
};

} // namespace

std::unique_ptr<Traversal> make_vulkan(std::span<const SurfaceGeometry> surfaces) {
    auto traversal = std::make_unique<VulkanTraversal>();
    traversal->initialize(surfaces);
    return traversal;
}

} // namespace raystrack
