#pragma once
#include "point_cloud_erosion.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <chrono>


typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef K::Point_3                                           Point;
typedef CGAL::Point_set_3<Point>                             Point_set;

// ─── EroderSetup ─────────────────────────────────────────────────────────
struct EroderSetup {
    ErosionConfig                    cfg;
    std::vector<std::array<float,3>> se_offsets;
    std::vector<std::array<float,3>> input_vec;
};

static EroderSetup
make_eroder_setup(const Point_set& data, const Point_set& se, double min_dist)
{
    std::vector<std::array<float,3>> input_vec;
    input_vec.reserve(data.size());
    for (auto it = data.begin(); it != data.end(); ++it) {
        const Point& p = data.point(*it);
        input_vec.push_back({ (float)p.x(), (float)p.y(), (float)p.z() });
    }

    std::vector<std::array<float,3>> se_offsets;
    se_offsets.reserve(se.size());
    for (auto it = se.begin(); it != se.end(); ++it) {
        const Point& p = se.point(*it);
        se_offsets.push_back({ (float)p.x(), (float)p.y(), (float)p.z() });
    }

    const uint32_t n = (uint32_t)input_vec.size();
    auto next_pow2 = [](uint32_t v) -> uint32_t {
        uint32_t t = 1u; while (t < v) t <<= 1; return t;
    };
    const uint32_t table_size = next_pow2(n * 2);

    std::cout << "[erode] n=" << n
              << "  table_size=" << table_size
              << "  GPU: hash=" << (uint64_t)table_size * 16 / (1024*1024) << " MB"
              << "  input="     << (uint64_t)n * 16          / (1024*1024) << " MB\n";

    ErosionConfig cfg;
    cfg.cell_size  = static_cast<float>(min_dist);
    cfg.min_dist   = static_cast<float>(min_dist);
    cfg.table_size = table_size;
    cfg.max_input  = n;
    cfg.max_output = n;

    return { cfg, se_offsets, input_vec };
}

// ─── erode ────────────────────────────────────────────────────────────────
Point_set erode(const Point_set& data, const Point_set& se, double min_dist)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    if (data.empty() || se.empty()) return Point_set{};
    auto [cfg, se_offsets, input_vec] = make_eroder_setup(data, se, min_dist);
    PointCloudEroder eroder(cfg);
    eroder.set_structuring_element(se_offsets);
    eroder.erode(input_vec);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Total time  = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    return eroder.get_result_cgal();
}

// ─── erode_score ──────────────────────────────────────────────────────────
Point_set erode_score(const Point_set& data, const Point_set& se, double min_dist)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    if (data.empty() || se.empty()) return Point_set{};
    auto [cfg, se_offsets, input_vec] = make_eroder_setup(data, se, min_dist);
    PointCloudEroder eroder(cfg);
    eroder.set_structuring_element(se_offsets);
    eroder.erode_score(input_vec);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Total time  = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    return eroder.get_result_cgal_with_scores();
}