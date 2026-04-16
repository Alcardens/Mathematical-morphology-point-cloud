#pragma once
#include "point_cloud_erosion.h"
#include "point_cloud_erosion_density.h"
#include "point_cloud_erosion_orientation.h"
#include "density.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <chrono>


typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef K::Point_3                                           Point;
typedef K::Vector_3                                          Vector;
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
    cfg.chunk_size = std::min(n, 500'000u);

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

// ─── DensityEroderSetup ───────────────────────────────────────────────────
struct DensityEroderSetup {
    DensityErosionConfig             cfg;
    std::vector<std::array<float,4>> se_offsets;
    std::vector<std::array<float,4>> input_vec;
};

static DensityEroderSetup
make_density_eroder_setup(const Point_set& data_with_density,
                          const Point_set& se,
                          float min_dens, float max_dens, float increment)
{
    // ── Convert input points (xyz + density from property map) ────────────
    auto density_opt = data_with_density.property_map<float>("density");

    std::vector<std::array<float,4>> input_vec;
    input_vec.reserve(data_with_density.size());
    for (auto it = data_with_density.begin(); it != data_with_density.end(); ++it) {
        const Point& p = data_with_density.point(*it);
        float d = density_opt.has_value() ? (*density_opt)[*it] : min_dens;
        input_vec.push_back({ (float)p.x(), (float)p.y(), (float)p.z(), d });
    }

    // ── Convert SE offsets (xyz + density from property map) ─────────────
    auto se_density_opt = se.property_map<float>("density");

    std::vector<std::array<float,4>> se_offsets;
    se_offsets.reserve(se.size());
    for (auto it = se.begin(); it != se.end(); ++it) {
        const Point& p = se.point(*it);
        float d = se_density_opt.has_value() ? (*se_density_opt)[*it] : min_dens;
        se_offsets.push_back({ (float)p.x(), (float)p.y(), (float)p.z(), d });
    }

    const uint32_t n = (uint32_t)input_vec.size();
    auto next_pow2 = [](uint32_t v) -> uint32_t {
        uint32_t t = 1u; while (t < v) t <<= 1; return t;
    };
    const uint32_t table_size = next_pow2(n * 2);
    const uint32_t chunk_size = std::min(n, 500'000u);

    std::cout << "[erode_density] n=" << n
              << "  table_size=" << table_size
              << "  density range=[" << min_dens << ", " << max_dens << "]"
              << "  increment=" << increment << "\n"
              << "  GPU: hash=" << (uint64_t)table_size * 16 / (1024*1024) << " MB"
              << "  input="     << (uint64_t)n * 16          / (1024*1024) << " MB\n";

    DensityErosionConfig cfg;
    cfg.min_dens   = min_dens;
    cfg.max_dens   = max_dens;
    cfg.increment  = increment;
    cfg.table_size = table_size;
    cfg.max_input  = n;
    cfg.max_output = n;
    cfg.chunk_size = chunk_size;
    cfg.max_shell  = (int)std::ceil(max_dens / min_dens) + 1;

    return { cfg, se_offsets, input_vec };
}

// ─── erode_density ────────────────────────────────────────────────────────
Point_set erode_density(const Point_set& data, const Point_set& se, float increment)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    if (data.empty() || se.empty()) return Point_set{};

    // ── 1. Estimate density of input ──────────────────────────────────────
    std::cout << "[erode_density] estimating density...\n";
    Point_set data_with_density = density_estimate(data);

    // ── 2. Compute density range from input ───────────────────────────────
    auto density_opt = data_with_density.property_map<float>("density");
    float min_dens =  std::numeric_limits<float>::max();
    float max_dens = -std::numeric_limits<float>::max();
    for (auto it = data_with_density.begin(); it != data_with_density.end(); ++it) {
        float d = (*density_opt)[*it];
        min_dens = std::min(min_dens, d);
        max_dens = std::max(max_dens, d);
    }

    // ── 3. Setup and run ──────────────────────────────────────────────────
    auto [cfg, se_offsets, input_vec] =
        make_density_eroder_setup(data_with_density, se, min_dens, max_dens, increment);

    PointCloudDensityEroder eroder(cfg);
    eroder.set_structuring_element(se_offsets);
    eroder.erode(input_vec);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Total time = "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " [ms]\n";

    return eroder.get_result_cgal();
}

// ─── erode_density_score ──────────────────────────────────────────────────
Point_set erode_density_score(const Point_set& data, const Point_set& se, float increment)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    if (data.empty() || se.empty()) return Point_set{};

    // ── 1. Estimate density of input ──────────────────────────────────────
    std::cout << "[erode_density_score] estimating density...\n";
    Point_set data_with_density = density_estimate(data);

    // ── 2. Compute density range from input ───────────────────────────────
    auto density_opt = data_with_density.property_map<float>("density");
    float min_dens =  std::numeric_limits<float>::max();
    float max_dens = -std::numeric_limits<float>::max();
    for (auto it = data_with_density.begin(); it != data_with_density.end(); ++it) {
        float d = (*density_opt)[*it];
        min_dens = std::min(min_dens, d);
        max_dens = std::max(max_dens, d);
    }

    // ── 3. Setup and run ──────────────────────────────────────────────────
    auto [cfg, se_offsets, input_vec] =
        make_density_eroder_setup(data_with_density, se, min_dens, max_dens, increment);

    PointCloudDensityEroder eroder(cfg);
    eroder.set_structuring_element(se_offsets);
    eroder.erode_score(input_vec);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Total time = "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
              << " [ms]\n";

    return eroder.get_result_cgal_with_scores();
}

struct OrientationEroderSetup {
    OrientationErosionConfig         cfg;
    std::vector<std::array<float,3>> se_offsets;
    std::vector<std::array<float,3>> input_vec;
    std::vector<std::array<float,3>> normals;
};

static OrientationEroderSetup
make_orientation_eroder_setup(const Point_set& data, const Point_set& se, double min_dist)
{
    auto norm_opt = data.property_map<Vector>("normal");

    std::vector<std::array<float,3>> input_vec;
    std::vector<std::array<float,3>> normals;
    input_vec.reserve(data.size());
    for (auto it = data.begin(); it != data.end(); ++it) {
        const Point& p = data.point(*it);
        const Vector& norm = (*norm_opt)[*it];
        input_vec.push_back({ (float)p.x(), (float)p.y(), (float)p.z() });
        normals.push_back({ (float)norm.x(), (float)norm.y(), (float)norm.z() });
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

    OrientationErosionConfig cfg;
    cfg.cell_size  = static_cast<float>(min_dist);
    cfg.min_dist   = static_cast<float>(min_dist);
    cfg.table_size = table_size;
    cfg.max_input  = n;
    cfg.max_output = n;
    cfg.chunk_size = std::min(n, 500'000u);

    return { cfg, se_offsets, input_vec, normals };
}

// ─── erode ────────────────────────────────────────────────────────────────
Point_set erode_orientation(const Point_set& data, const Point_set& se, double min_dist)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    if (data.empty() || se.empty()) return Point_set{};
    auto [cfg, se_offsets, input_vec, normals] = make_orientation_eroder_setup(data, se, min_dist);
    PointCloudOrientationEroder eroder(cfg);
    eroder.set_structuring_element(se_offsets);
    eroder.erode(input_vec, normals);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Total time  = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    return eroder.get_result_cgal();
}

// ─── erode_score ──────────────────────────────────────────────────────────
Point_set erode_orientation_score(const Point_set& data, const Point_set& se, double min_dist)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    if (data.empty() || se.empty()) return Point_set{};
    auto [cfg, se_offsets, input_vec, normals] = make_orientation_eroder_setup(data, se, min_dist);
    PointCloudOrientationEroder eroder(cfg);
    eroder.set_structuring_element(se_offsets);
    eroder.erode_score(input_vec, normals);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Total time  = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    return eroder.get_result_cgal_with_scores();
}
