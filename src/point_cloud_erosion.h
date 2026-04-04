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

struct ErosionConfig {
    float    cell_size;
    float    min_dist;
    uint32_t table_size;
    uint32_t max_input;
    uint32_t max_output;
};

class PointCloudEroder {
public:
    explicit PointCloudEroder(const ErosionConfig& cfg);
    ~PointCloudEroder();

    void set_structuring_element(const std::vector<std::array<float, 3>>& se);

    // Upload all input points, build the input hash, run the erosion shader.
    // Returns the number of surviving points.
    uint32_t erode(const std::vector<std::array<float, 3>>& input);

    uint32_t erode_score(const std::vector<std::array<float, 3>>& input);

    std::vector<std::array<float, 3>> get_result()      const;
    Point_set                         get_result_cgal() const;

    std::vector<std::array<float, 4>> get_result_with_scores() const;
    Point_set                         get_result_cgal_with_scores() const;

private:
    ErosionConfig m_cfg;
    uint32_t      m_result_count = 0;
    uint32_t      m_se_count     = 0;

    uint32_t dispatch_erode(const std::vector<std::array<float, 3>>& input,
                        GLuint prog, bool write_all);

    // Input hash — stores all input points
    GLuint m_ssbo_input_cells  = 0;   // open-addressing table: uint[table_size * 2]
    GLuint m_ssbo_input_pts    = 0;   // float4[max_input]

    // Structuring element
    GLuint m_ssbo_se           = 0;   // float4[se_count]

    // Output
    GLuint m_ssbo_out_pts      = 0;   // float4[max_output]
    GLuint m_ssbo_out_count    = 0;   // uint[1]

    GLuint m_prog_insert = 0;
    GLuint m_prog_erode  = 0;
    GLuint m_prog_erode_score = 0;

    static GLuint compile_compute(const char* src);
    void build_input_hash(uint32_t point_count);
};