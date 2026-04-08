#pragma once
#include "point_cloud_addition.h"
#include "point_cloud_intersection.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef K::Point_3                                           Point;
typedef CGAL::Point_set_3<Point>                             Point_set;

// ─── add ──────────────────────────────────────────────────────────────────
Point_set add(const Point_set& original, const Point_set& addition, double min_dist)
{
    if (original.empty()) return addition;
    if (addition.empty()) return original;

    // ── Convert to flat arrays ────────────────────────────────────────────
    auto to_vec = [](const Point_set& ps) {
        std::vector<std::array<float, 3>> out;
        out.reserve(ps.size());
        for (auto it = ps.begin(); it != ps.end(); ++it) {
            const Point& p = ps.point(*it);
            out.push_back({ (float)p.x(), (float)p.y(), (float)p.z() });
        }
        return out;
    };

    auto orig_vec = to_vec(original);
    auto add_vec  = to_vec(addition);

    const uint32_t n_orig = (uint32_t)orig_vec.size();
    const uint32_t n_add  = (uint32_t)add_vec.size();

    // ── Size GPU buffers ──────────────────────────────────────────────────
    auto next_pow2 = [](uint32_t v) -> uint32_t {
        uint32_t t = 1u; while (t < v) t <<= 1; return t;
    };
    // Hash must fit originals + all possible accepted additions
    const uint32_t max_total  = n_orig + n_add;
    const uint32_t table_size = next_pow2(max_total * 2);

    std::cout << "[add] original=" << n_orig
              << "  addition=" << n_add
              << "  table_size=" << table_size
              << "  GPU: hash=" << (uint64_t)table_size * 16 / (1024*1024) << " MB\n";

    AdditionConfig cfg;
    cfg.cell_size    = static_cast<float>(min_dist);
    cfg.min_dist     = static_cast<float>(min_dist);
    cfg.table_size   = table_size;
    cfg.num_original = n_orig;
    cfg.num_addition = n_add;

    // ── Run ───────────────────────────────────────────────────────────────
    PointCloudAdder adder(cfg);
    adder.add(orig_vec, add_vec);

    // ── Pack result into Point_set ────────────────────────────────────────
    auto pts = adder.get_result();
    Point_set result;
    for (auto& p : pts)
        result.insert(Point(p[0], p[1], p[2]));
    return result;
}

Point_set intersect(const Point_set& original, const Point_set& intersection, double min_dist)
{
  if ((original.empty()) || (intersection.empty())) return Point_set{};

    // ── Convert to flat arrays ────────────────────────────────────────────
    auto to_vec = [](const Point_set& ps) {
        std::vector<std::array<float, 3>> out;
        out.reserve(ps.size());
        for (auto it = ps.begin(); it != ps.end(); ++it) {
            const Point& p = ps.point(*it);
            out.push_back({ (float)p.x(), (float)p.y(), (float)p.z() });
        }
        return out;
    };

    auto orig_vec = to_vec(original);
    auto isec_vec = to_vec(intersection);

    const uint32_t n_orig = (uint32_t)orig_vec.size();
    const uint32_t n_isec  = (uint32_t)isec_vec.size();

    // ── Size GPU buffers ──────────────────────────────────────────────────
    auto next_pow2 = [](uint32_t v) -> uint32_t {
        uint32_t t = 1u; while (t < v) t <<= 1; return t;
    };
    // Hash must fit originals + all possible accepted additions
    const uint32_t max_total  = n_orig;
    const uint32_t table_size = next_pow2(n_isec * 2);

    std::cout << "[intersect] original=" << n_orig
              << "  intersection=" << n_isec
              << "  table_size=" << table_size
              << "  GPU: hash=" << (uint64_t)table_size * 16 / (1024*1024) << " MB\n";

    IntersectionConfig cfg;
    cfg.cell_size    = static_cast<float>(min_dist);
    cfg.min_dist     = static_cast<float>(min_dist);
    cfg.table_size   = table_size;
    cfg.num_original = n_orig;
    cfg.num_intersection = n_isec;

    // ── Run ───────────────────────────────────────────────────────────────
    PointCloudIntersector intersector(cfg);
    intersector.intersect(orig_vec, isec_vec);

    // ── Pack result into Point_set ────────────────────────────────────────
    auto pts = intersector.get_result();
    Point_set result;
    for (auto& p : pts)
        result.insert(Point(p[0], p[1], p[2]));
    return result;
}

Point_set subtract(const Point_set& original, const Point_set& intersection, double min_dist)
{
  if ((original.empty()) || (intersection.empty())) return Point_set{};

    // ── Convert to flat arrays ────────────────────────────────────────────
    auto to_vec = [](const Point_set& ps) {
        std::vector<std::array<float, 3>> out;
        out.reserve(ps.size());
        for (auto it = ps.begin(); it != ps.end(); ++it) {
            const Point& p = ps.point(*it);
            out.push_back({ (float)p.x(), (float)p.y(), (float)p.z() });
        }
        return out;
    };

    auto orig_vec = to_vec(original);
    auto isec_vec = to_vec(intersection);

    const uint32_t n_orig = (uint32_t)orig_vec.size();
    const uint32_t n_isec  = (uint32_t)isec_vec.size();

    // ── Size GPU buffers ──────────────────────────────────────────────────
    auto next_pow2 = [](uint32_t v) -> uint32_t {
        uint32_t t = 1u; while (t < v) t <<= 1; return t;
    };
    // Hash must fit originals + all possible accepted additions
    const uint32_t max_total  = n_orig;
    const uint32_t table_size = next_pow2(n_isec * 2);

    std::cout << "[intersect] original=" << n_orig
              << "  intersection=" << n_isec
              << "  table_size=" << table_size
              << "  GPU: hash=" << (uint64_t)table_size * 16 / (1024*1024) << " MB\n";

    IntersectionConfig cfg;
    cfg.cell_size    = static_cast<float>(min_dist);
    cfg.min_dist     = static_cast<float>(min_dist);
    cfg.table_size   = table_size;
    cfg.num_original = n_orig;
    cfg.num_intersection = n_isec;

    // ── Run ───────────────────────────────────────────────────────────────
    PointCloudIntersector intersector(cfg);
    intersector.subtract(orig_vec, isec_vec);

    // ── Pack result into Point_set ────────────────────────────────────────
    auto pts = intersector.get_result();
    Point_set result;
    for (auto& p : pts)
        result.insert(Point(p[0], p[1], p[2]));
    return result;
}