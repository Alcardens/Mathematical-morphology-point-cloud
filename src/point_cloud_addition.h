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

struct AdditionConfig {
    float    cell_size;
    float    min_dist;
    uint32_t table_size;      // power-of-two, >= 2 * max_original
    uint32_t num_original;    // max points in the original set
    uint32_t num_addition;    // max points in the addition set
};

class PointCloudAdder {
public:
    explicit PointCloudAdder(const AdditionConfig& cfg);
    ~PointCloudAdder();

    uint32_t add(const std::vector<std::array<float, 3>>& original,
             const std::vector<std::array<float, 3>>& addition);

    uint32_t result_count() const { return m_result_count; }

    std::vector<std::array<float, 3>> get_result()      const;
    Point_set                         get_result_cgal() const;

private:
    AdditionConfig m_cfg;
    uint32_t       m_result_count = 0;

    // Hash of the original point set — used by far_enough in the shader
    GPUSpatialHash m_hash;

    GLuint m_ssbo_addition_pts = 0;  // float4[max_addition]
    GLuint m_ssbo_out_pts      = 0;  // float4[max_output]
    GLuint m_ssbo_out_count    = 0;  // uint[1]

    GLuint m_prog_add = 0;

    static GLuint compile_compute(const char* src);
};