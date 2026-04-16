#pragma once
#include "gpu_spatial_hash.h"       // GPUSpatialHash, SpatialHashConfig
#include "point_cloud_dilation.h"   // PointCloudDilator, DilationConfig
#include "point_cloud_dilation_density.h"
#include "point_cloud_dilation_orientation.h"
#include "grouping.h"
#include "density.h"

#include <chrono>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/IO/write_ply_points.h>
#include <CGAL/IO/read_ply_points.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef K::Point_3                                           Point;
typedef K::Vector_3                                          Vector;
typedef CGAL::Point_set_3<Point>                             Point_set;

// ─── Helper: convert CGAL Point_set → flat float array ────────────────────
static std::vector<std::array<float, 3>>
point_set_to_vec(const Point_set& ps)
{
    std::vector<std::array<float, 3>> out;
    out.reserve(ps.size());
    for (auto it = ps.begin(); it != ps.end(); ++it) {
        const Point& p = ps.point(*it);
        out.push_back({ (float)p.x(), (float)p.y(), (float)p.z() });
    }
    return out;
}

Point_set dilate(const Point_set& data,
                 const Point_set& se,
                 double           min_dist)
{
    if (data.empty() || se.empty())
        return Point_set{};

    // ── 1. Convert structuring element to offset vectors ──────────────────
    // SE points are used as offsets
    std::vector<std::array<float, 3>> se_offsets;
    se_offsets.reserve(se.size());
    for (auto it = se.begin(); it != se.end(); ++it) {
        const Point& p = se.point(*it);
        se_offsets.push_back({
            (float)(p.x()),
            (float)(p.y()),
            (float)(p.z())
        });
    }

    // ── 2. Estimate SE diameter (bounding-box diagonal) ───────────────────
    float se_min[3] = { se_offsets[0][0], se_offsets[0][1], se_offsets[0][2] };
    float se_max[3] = { se_offsets[0][0], se_offsets[0][1], se_offsets[0][2] };
    for (auto& o : se_offsets)
        for (int i = 0; i < 3; ++i) {
            se_min[i] = std::min(se_min[i], o[i]);
            se_max[i] = std::max(se_max[i], o[i]);
        }
    double se_diameter =
        std::sqrt(std::pow(se_max[0]-se_min[0], 2) +
                  std::pow(se_max[1]-se_min[1], 2) +
                  std::pow(se_max[2]-se_min[2], 2));

    // ── 3. Split input into spatial groups ────────────────────────────────
    // split_by_se_diameter uses D as the characteristic grouping scale.

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    std::vector<Point_set> groups = split_by_se_diameter(data, se_diameter +  min_dist);

    if (groups.empty())
        return Point_set{};

    // ── 4. Size the GPU buffers ───────────────────────────────────────────
    // max_output: cap total output points at 50M or the theoretical maximum
    // (one output point per min_dist^3 voxel in the input bounding box).
    const uint32_t max_output = static_cast<uint32_t>(
        std::min((size_t)50'000'000,
                 data.size() * se.size() + data.size()));

    size_t max_group_size = 0;
    for (auto& g : groups) max_group_size = std::max(max_group_size, g.size());
    const uint32_t max_candidates = static_cast<uint32_t>(
        std::min((size_t)1'000'000, max_group_size * se.size()));

    // Open-addressing table: 2 uints per slot, table_size >= 2*max_output.
    // Memory = table_size * 8 bytes — no per-cell chain needed.
    auto next_pow2 = [](uint32_t v) -> uint32_t {
        uint32_t t = 1u; while (t < v) t <<= 1; return t;
    };
    const uint32_t table_size = next_pow2(max_output * 2);
    const uint64_t cells_mb   = (uint64_t)table_size * 8          / (1024*1024);
    const uint64_t points_mb  = (uint64_t)max_output * 4*4        / (1024*1024);
    std::cout << "[dilate] max_output=" << max_output
              << "  table_size=" << table_size
              << "  GPU: cells=" << cells_mb << " MB"
              << "  points=" << points_mb << " MB\n";

    DilationConfig cfg;
    cfg.cell_size        = static_cast<float>(min_dist);
    cfg.min_dist         = static_cast<float>(min_dist);
    cfg.table_size       = table_size;
    cfg.max_total_points = max_output;
    cfg.max_candidates   = max_candidates;
    cfg.max_input_group  = static_cast<uint32_t>(max_group_size);


    // ── 5. Run dilation across all groups ─────────────────────────────────
    PointCloudDilator dilator(cfg);
    dilator.set_structuring_element(se_offsets);


    std::cout << "[dilate] " << groups.size() << " groups, "
              << se_offsets.size() << " SE offsets, "
              << "min_dist=" << min_dist << "\n";

    std::chrono::steady_clock::time_point algo_begin = std::chrono::steady_clock::now();

    for (size_t gi = 0; gi < groups.size(); ++gi) {
        const Point_set& grp = groups[gi];
        if (grp.empty()) continue;

        auto pts = point_set_to_vec(grp);
        dilator.process_group(pts);

        // std::cout << "  group " << gi << "/" << groups.size()
        //           << "  size=" << grp.size()
        //           << "  total_out=" << dilator.output_point_count() << "\n";
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Total time  = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    std::cout << "Dilate time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - algo_begin).count() << "[ms]" << std::endl;

    // ── 6. Download result and pack into a CGAL Point_set ─────────────────
    return dilator.get_result_cgal();
}

static std::vector<std::array<float, 4>>
point_set_to_vec4(const Point_set& ps, const std::string& prop_name)
{
    auto prop_opt = ps.property_map<float>(prop_name);
    bool has_prop = prop_opt.has_value();

    std::vector<std::array<float, 4>> out;
    out.reserve(ps.size());
    for (auto it = ps.begin(); it != ps.end(); ++it) {
        const Point& p = ps.point(*it);
        float w = has_prop ? (*prop_opt)[*it] : 0.f;
        out.push_back({ (float)p.x(), (float)p.y(), (float)p.z(), w });
    }
    return out;
}

Point_set dilate_density(const Point_set& data, const Point_set& se, const float increment)
{
    if (data.empty() || se.empty())
        return Point_set{};

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    // ── 1. Estimate density of the input point cloud ──────────────────────
    std::cout << "[dilate_density] estimating density...\n";
    Point_set data_with_density = density_estimate(data);

    // ── 2. Extract SE offsets as float[4] (xyz offset + density in w) ─────
    auto se_density_opt = se.property_map<float>("density");
    bool se_has_density = se_density_opt.has_value();

    std::vector<std::array<float, 4>> se_offsets;
    se_offsets.reserve(se.size());
    for (auto it = se.begin(); it != se.end(); ++it) {
        const Point& p = se.point(*it);
        float d = se_has_density ? (*se_density_opt)[*it] : 0.f;
        se_offsets.push_back({ (float)p.x(), (float)p.y(), (float)p.z(), d });
    }

    // ── 3. Compute density range from SE ─────────────────────────
    auto data_density_opt = data_with_density.property_map<float>("density");
    float min_dens =  std::numeric_limits<float>::max();
    float max_dens = -std::numeric_limits<float>::max();


    for (auto& s : se_offsets) {
        min_dens = std::min(min_dens, s[3]);
        max_dens = std::max(max_dens, s[3]);
    }

    std::cout << "[dilate_density] density range: [" << min_dens << ", " << max_dens << "]"
              << "  increment=" << increment << "\n";

    // ── 4. Compute SE bounding-box diameter for grouping ──────────────────
    float se_min[3] = { se_offsets[0][0], se_offsets[0][1], se_offsets[0][2] };
    float se_max[3] = { se_offsets[0][0], se_offsets[0][1], se_offsets[0][2] };
    for (auto& o : se_offsets)
        for (int i = 0; i < 3; ++i) {
            se_min[i] = std::min(se_min[i], o[i]);
            se_max[i] = std::max(se_max[i], o[i]);
        }
    double se_diameter = std::sqrt(
        std::pow(se_max[0]-se_min[0], 2) +
        std::pow(se_max[1]-se_min[1], 2) +
        std::pow(se_max[2]-se_min[2], 2));

    // ── 5. Split input into groups, carrying the density property ─────────
    std::vector<Point_set> groups =
        split_by_se_diameter(data_with_density, se_diameter + min_dens, "density");

    if (groups.empty()) return Point_set{};

    // ── 6. Size GPU buffers ───────────────────────────────────────────────
    const uint32_t max_output = static_cast<uint32_t>(
        std::min((size_t)50'000'000,
                 data.size() * se.size() + data.size()));

    size_t max_group_size = 0;
    for (auto& g : groups)
        max_group_size = std::max(max_group_size, g.size());

    const uint32_t max_candidates = static_cast<uint32_t>(
        std::min((size_t)1'000'000, max_group_size * se.size()));

    auto next_pow2 = [](uint32_t v) -> uint32_t {
        uint32_t t = 1u; while (t < v) t <<= 1; return t;
    };
    const uint32_t table_size = next_pow2(max_output * 2);

    std::cout << "[dilate_density] max_output=" << max_output
              << "  table_size=" << table_size
              << "  groups=" << groups.size()
              << "  se_offsets=" << se_offsets.size() << "\n";

    std::chrono::steady_clock::time_point algo_begin = std::chrono::steady_clock::now();

    // ── 7. Configure and run density-aware dilation ───────────────────────
    DensityDilationConfig cfg;
    cfg.min_dens        = min_dens;
    cfg.max_dens        = max_dens;
    cfg.increment       = increment;
    cfg.table_size      = table_size;
    cfg.max_total_points = max_output;
    cfg.max_candidates  = max_candidates;
    cfg.max_input_group = static_cast<uint32_t>(max_group_size);

    PointCloudDensityDilator dilator(cfg);
    dilator.set_structuring_element(se_offsets);

    for (size_t gi = 0; gi < groups.size(); ++gi) {
        const Point_set& grp = groups[gi];
        if (grp.empty()) continue;

        auto pts = point_set_to_vec4(grp, "density");
        dilator.process_group(pts);

        // std::cout << "  group " << gi << "/" << groups.size()
        //           << "  size=" << grp.size()
        //           << "  total_out=" << dilator.output_point_count() << "\n";
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Total time  = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    std::cout << "Dilate time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - algo_begin).count() << "[ms]" << std::endl;

    // ── 8. Download result — output has density in w ──────────────────────
    return dilator.get_result_cgal();
}

static std::vector<std::array<float, 3>>
point_set_to_normals(const Point_set& ps)
{
    std::vector<std::array<float, 3>> out;
    out.reserve(ps.size());

    auto norm_opt = ps.property_map<Vector>("normal");

    for (auto it = ps.begin(); it != ps.end(); ++it) {
        if (norm_opt.has_value()) {
            const Vector& n = (*norm_opt)[*it];
            out.push_back({ (float)n.x(), (float)n.y(), (float)n.z() });
        } else {
            out.push_back({ 0.f, 1.f, 0.f });
        }
    }
    return out;
}

Point_set dilate_orientation(const Point_set& data,
                             const Point_set& se,
                             double           min_dist)
{
    if (data.empty() || se.empty())
        return Point_set{};

    const bool has_normals = data.property_map<Vector>("normal").has_value();

    // ── 1. Convert SE to offset vectors ───────────────────────────────────
    std::vector<std::array<float, 3>> se_offsets;
    se_offsets.reserve(se.size());
    for (auto it = se.begin(); it != se.end(); ++it) {
        const Point& p = se.point(*it);
        se_offsets.push_back({ (float)p.x(), (float)p.y(), (float)p.z() });
    }

    // ── 2. SE bounding-box diameter ───────────────────────────────────────
    float se_min[3] = { se_offsets[0][0], se_offsets[0][1], se_offsets[0][2] };
    float se_max[3] = { se_offsets[0][0], se_offsets[0][1], se_offsets[0][2] };
    for (auto& o : se_offsets)
        for (int i = 0; i < 3; ++i) {
            se_min[i] = std::min(se_min[i], o[i]);
            se_max[i] = std::max(se_max[i], o[i]);
        }
    double se_diameter = std::sqrt(
        std::pow(se_max[0]-se_min[0], 2) +
        std::pow(se_max[1]-se_min[1], 2) +
        std::pow(se_max[2]-se_min[2], 2));

    // ── 3. Split input into spatial groups, carrying normals if present ────
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    std::vector<Point_set> groups =
        split_by_se_diameter(data, se_diameter + min_dist, has_normals);

    if (groups.empty()) return Point_set{};

    // ── 4. Size GPU buffers ───────────────────────────────────────────────
    const uint32_t max_output = static_cast<uint32_t>(
        std::min((size_t)50'000'000, data.size() * se.size() + data.size()));

    size_t max_group_size = 0;
    for (auto& g : groups) max_group_size = std::max(max_group_size, g.size());

    const uint32_t max_candidates = static_cast<uint32_t>(
        std::min((size_t)1'000'000, max_group_size * se.size()));

    auto next_pow2 = [](uint32_t v) -> uint32_t {
        uint32_t t = 1u; while (t < v) t <<= 1; return t;
    };
    const uint32_t table_size = next_pow2(max_output * 2);

    std::cout << "[dilate_orientation] max_output=" << max_output
              << "  table_size=" << table_size
              << "  GPU: cells=" << (uint64_t)table_size * 16 / (1024*1024) << " MB"
              << "  points=" << (uint64_t)max_output * 16 / (1024*1024) << " MB"
              << "  normals=" << (has_normals ? "yes" : "no") << "\n";

    // ── 5. Run dilation ───────────────────────────────────────────────────
    std::chrono::steady_clock::time_point algo_begin = std::chrono::steady_clock::now();

    if (has_normals) {
        // Orientation-aware dilation
        OrientationDilationConfig cfg;
        cfg.cell_size        = static_cast<float>(min_dist);
        cfg.min_dist         = static_cast<float>(min_dist);
        cfg.table_size       = table_size;
        cfg.max_total_points = max_output;
        cfg.max_candidates   = max_candidates;
        cfg.max_input_group  = static_cast<uint32_t>(max_group_size);

        PointCloudOrientationDilator dilator(cfg);
        dilator.set_structuring_element(se_offsets);

        std::cout << "[dilate_orientation] " << groups.size() << " groups, "
                  << se_offsets.size() << " SE offsets, min_dist=" << min_dist << "\n";

        for (size_t gi = 0; gi < groups.size(); ++gi) {
            const Point_set& grp = groups[gi];
            if (grp.empty()) continue;

            auto pts     = point_set_to_vec(grp);
            auto normals = point_set_to_normals(grp);
            dilator.process_group(pts, normals);
        }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Total time  = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()      << " [ms]\n";
        std::cout << "Dilate time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - algo_begin).count() << " [ms]\n";

        return dilator.get_result_cgal();

    } else {
        DilationConfig cfg;
        cfg.cell_size        = static_cast<float>(min_dist);
        cfg.min_dist         = static_cast<float>(min_dist);
        cfg.table_size       = table_size;
        cfg.max_total_points = max_output;
        cfg.max_candidates   = max_candidates;
        cfg.max_input_group  = static_cast<uint32_t>(max_group_size);

        PointCloudDilator dilator(cfg);
        dilator.set_structuring_element(se_offsets);

        std::cout << "[dilate_orientation] " << groups.size() << " groups, "
                  << se_offsets.size() << " SE offsets, min_dist=" << min_dist
                  << " (no normals — using standard dilation)\n";

        for (size_t gi = 0; gi < groups.size(); ++gi) {
            const Point_set& grp = groups[gi];
            if (grp.empty()) continue;
            dilator.process_group(point_set_to_vec(grp));
        }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Total time  = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()      << " [ms]\n";
        std::cout << "Dilate time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - algo_begin).count() << " [ms]\n";

        return dilator.get_result_cgal();
    }
}

