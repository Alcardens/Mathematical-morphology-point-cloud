#pragma once
#include "density_estimation.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef K::Point_3                                           Point;
typedef CGAL::Point_set_3<Point>                             Point_set;
typedef CGAL::Search_traits_3<K>                             Traits;
typedef CGAL::Kd_tree<Traits>                                Tree;
typedef CGAL::Orthogonal_k_neighbor_search<Traits>           KNN;

static float estimate_cell_size(const std::vector<std::array<float, 3>>& pts,
                                uint32_t sample_size = 500,
                                int      k           = 2)  // k=2: self + 1 neighbour
{
    const uint32_t n = (uint32_t)pts.size();
    if (n < 2) return 1.0f;

    // Build CGAL points vector
    std::vector<Point> cgal_pts;
    cgal_pts.reserve(n);
    for (auto& p : pts)
        cgal_pts.emplace_back(p[0], p[1], p[2]);

    // Build kd-tree on the full set
    Tree tree(cgal_pts.begin(), cgal_pts.end());

    // Draw a random sample to query
    sample_size = std::min(sample_size, n);
    std::vector<uint32_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0u);
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(sample_size);

    // Average distance to the nearest non-self neighbour
    double sum = 0.0;
    for (uint32_t idx : indices) {
        KNN search(tree, cgal_pts[idx], k);
        auto it = search.begin();
        ++it;  // skip self (distance 0)
        sum += std::sqrt(CGAL::to_double(it->second));
    }

    return static_cast<float>(sum / sample_size);
}

// ─── density_estimate ────────────────────────────────────────────────────
Point_set density_estimate(const Point_set& data)
{
    if (data.empty()) return Point_set{};

    // ── 1. Convert to flat array ──────────────────────────────────────────
    std::vector<std::array<float, 3>> pts;
    pts.reserve(data.size());
    for (auto it = data.begin(); it != data.end(); ++it) {
        const Point& p = data.point(*it);
        pts.push_back({ (float)p.x(), (float)p.y(), (float)p.z() });
    }

    const uint32_t n = (uint32_t)pts.size();

    // ── 2. Estimate cell size from overall point spacing ──────────────────
    const float cell_size = estimate_cell_size(pts);
    std::cout << "[density_estimate] n=" << n
              << "  estimated cell_size=" << cell_size << "\n";

    // ── 3. Size GPU buffers ───────────────────────────────────────────────
    auto next_pow2 = [](uint32_t v) -> uint32_t {
        uint32_t t = 1u; while (t < v) t <<= 1; return t;
    };
    const uint32_t table_size = next_pow2(n * 2);

    std::cout << "[density_estimate] table_size=" << table_size
              << "  GPU: hash=" << (uint64_t)table_size * 16 / (1024*1024) << " MB"
              << "  points=" << (uint64_t)n * 16 / (1024*1024) << " MB\n";

    DensityConfig cfg;
    cfg.table_size = table_size;
    cfg.cell_size  = cell_size;
    cfg.num_points = n;
    cfg.max_shell  = 6;  // 6 shells covers ~2500 cells — sufficient for K=5

    // ── 4. Run GPU density estimation ─────────────────────────────────────
    DensityEstimator estimator(cfg);
    estimator.density_estimation(pts);

    // ── 5. Pack result into Point_set with density property map ──────────
    return estimator.get_result_cgal();
}