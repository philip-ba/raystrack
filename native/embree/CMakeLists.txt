find_package(embree 4.4.1 EXACT CONFIG REQUIRED)

add_library(raystrack_embree STATIC traversal.cpp)
target_compile_features(raystrack_embree PRIVATE cxx_std_20)
target_include_directories(raystrack_embree PRIVATE ../include)
target_link_libraries(raystrack_embree PUBLIC embree)
set_target_properties(raystrack_embree PROPERTIES POSITION_INDEPENDENT_CODE ON)
