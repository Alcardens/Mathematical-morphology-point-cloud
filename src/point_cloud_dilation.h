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

struct DilationConfig {
    float    cell_size;
    float    min_dist;
    uint32_t table_size;        // power-of-two, >= 2 * max_total_points
    uint32_t max_total_points;
    uint32_t max_candidates;
    uint32_t max_input_group;
};

class PointCloudDilator {
public:
    explicit PointCloudDilator(const DilationConfig& cfg);
    ~PointCloudDilator();

    void set_structuring_element(const std::vector<std::array<float, 3>>& se);
    void process_group(const std::vector<std::array<float, 3>>& group);

    uint32_t output_point_count() const { return m_hash.point_count(); }
    bool     is_full()            const { return m_hash.is_full(); }

    std::vector<std::array<float, 3>>  get_result()      const;
    Point_set                          get_result_cgal() const;

private:
    void dispatch_dilation(uint32_t input_count);

    DilationConfig  m_cfg;
    GPUSpatialHash  m_hash;

    GLuint   m_ssbo_input    = 0;
    GLuint   m_ssbo_se       = 0;
    GLuint   m_ssbo_cand_pts = 0;
    GLuint   m_ssbo_cand_cnt = 0;
    GLuint   m_prog_dilate   = 0;
    uint32_t m_se_count      = 0;

    static GLuint compile_compute(const char* src);
};
