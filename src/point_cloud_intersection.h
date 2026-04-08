#pragma once
#include "gpu_spatial_hash.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <vector>
#include <array>
#include <cstdint>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3                                          Point;
typedef CGAL::Point_set_3<Point>                            Point_set;

struct IntersectionConfig {
    float    cell_size;
    float    min_dist;
    uint32_t table_size;       // power-of-two, >= 2 * max_original
    uint32_t num_original;     // max points in the original set
    uint32_t num_intersection; // max points in the intersection set
};

class PointCloudIntersector {
public:
    explicit PointCloudIntersector(const IntersectionConfig& cfg);
    ~PointCloudIntersector();

    uint32_t intersect(const std::vector<std::array<float, 3>>& original,
             const std::vector<std::array<float, 3>>& intersection);

    uint32_t subtract(const std::vector<std::array<float, 3>>& original,
             const std::vector<std::array<float, 3>>& subtract);

    uint32_t result_count() const { return m_result_count; }

    std::vector<std::array<float, 3>> get_result()      const;
    Point_set                         get_result_cgal() const;

private:
    IntersectionConfig m_cfg;
    uint32_t           m_result_count = 0;

    // Hash of the original point set — used by far_enough in the shader
    GPUSpatialHash m_hash;

    GLuint m_ssbo_original_pts = 0;  // float4[max_addition]
    GLuint m_ssbo_out_pts      = 0;  // float4[max_output]
    GLuint m_ssbo_out_count    = 0;  // uint[1]

    uint32_t dispatch_operation(const std::vector<std::array<float, 3>>& original,
                            const std::vector<std::array<float, 3>>& other,
                            GLuint prog, const char* label);

    GLuint m_prog_intersect = 0;
    GLuint m_prog_subtract = 0;

    static GLuint compile_compute(const char* src);
};
