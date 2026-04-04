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

struct DensityConfig {
    uint32_t  table_size;
    float     cell_size;
    uint32_t  num_points;
    uint32_t  max_shell;
};

class DensityEstimator {
public:
    explicit DensityEstimator(const DensityConfig& cfg);
    ~DensityEstimator();

    void density_estimation(const std::vector<std::array<float, 3>>& points);

    std::vector<std::array<float, 4>>  get_result()      const;
    Point_set                          get_result_cgal() const;

private:
    DensityConfig   m_cfg;
    GPUSpatialHash  m_hash;

    GLuint   m_ssbo_out_dist  = 0;   // float[num_points] — density value per point
    GLuint   m_prog_density   = 0;

    static GLuint compile_compute(const char* src);
};
