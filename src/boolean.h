#pragma once
#include "point_cloud_addition.h"
#include "point_cloud_intersection.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>

#include <string>
#include <vector>
#include <set>

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef K::Point_3                                           Point;
typedef CGAL::Point_set_3<Point>                             Point_set;
typedef CGAL::Search_traits_3<K>                             Traits;
typedef CGAL::Kd_tree<Traits>                                Tree;
typedef CGAL::Orthogonal_k_neighbor_search<Traits>           KNN;

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

// Collect property names of type T from a Point_set
template <typename T>
static std::vector<std::string> props_of(const Point_set& ps)
{
    std::vector<std::string> names;
    for (const std::string& name : ps.properties()) {
        if (name == "point") continue;
        if (ps.property_map<T>(name).has_value())
            names.push_back(name);
    }
    return names;
}

// Register, then copy one property type for one point
template <typename T>
static void handle_type(
    const Point_set&             copy_to,
    Point_set::Index             to_idx,
    const Point_set&             copy_from,
    Point_set::Index             from_idx,
    Point_set&                   result,
    Point_set::Index             dst_idx,
    const std::set<std::string>& names)
{
    for (const std::string& name : names) {
        auto dst_opt = result.property_map<T>(name);
        if (!dst_opt.has_value()) continue;

        auto from_opt = copy_from.property_map<T>(name);
        if (from_opt.has_value()) {
            (*dst_opt)[dst_idx] = (*from_opt)[from_idx];
            continue;
        }
        auto to_opt = copy_to.property_map<T>(name);
        if (to_opt.has_value())
            (*dst_opt)[dst_idx] = (*to_opt)[to_idx];
    }
}

// Register all names of type T on result
template <typename T>
static std::set<std::string> register_type(
    const Point_set& copy_to,
    const Point_set& copy_from,
    Point_set&       result,
    T                default_val)
{
    auto to_names   = props_of<T>(copy_to);
    auto from_names = props_of<T>(copy_from);
    std::set<std::string> names(to_names.begin(), to_names.end());
    for (auto& n : from_names) names.insert(n);
    for (const std::string& name : names)
        result.add_property_map<T>(name, default_val);
    return names;
}

// ─── copy_attributes ──────────────────────────────────────────────────────
Point_set copy_attributes(const Point_set& copy_to, const Point_set& copy_from)
{
    Point_set result;

    // ── 1. Register all property types on result ──────────────────────────
    auto float_names  = register_type<float>   (copy_to, copy_from, result, 0.f);
    auto double_names = register_type<double>  (copy_to, copy_from, result, 0.0);
    auto int_names    = register_type<int>     (copy_to, copy_from, result, 0);
    auto uint_names   = register_type<uint32_t>(copy_to, copy_from, result, 0u);
    auto uchar_names  = register_type<uint8_t> (copy_to, copy_from, result, uint8_t(0));
    auto vec_names    = register_type<Vector>  (copy_to, copy_from, result, Vector(0,1,0));

    // ── 2. Build KD-tree on copy_from ─────────────────────────────────────
    std::vector<Point> from_pts;
    std::vector<Point_set::Index> from_indices;
    from_pts.reserve(copy_from.size());
    from_indices.reserve(copy_from.size());
    for (auto it = copy_from.begin(); it != copy_from.end(); ++it) {
        from_pts.push_back(copy_from.point(*it));
        from_indices.push_back(*it);
    }
    Tree tree(from_pts.begin(), from_pts.end());

    // ── 3. For each point in copy_to, find nearest and copy all properties
    for (auto it = copy_to.begin(); it != copy_to.end(); ++it) {
        const Point& p = copy_to.point(*it);

        KNN search(tree, p, 1);
        // Resolve nearest point to its index
        const Point& nearest_pt = search.begin()->first;
        Point_set::Index nearest_idx;
        for (size_t i = 0; i < from_pts.size(); ++i) {
            if (from_pts[i] == nearest_pt) { nearest_idx = from_indices[i]; break; }
        }

        auto dst_it = result.insert(p);

        handle_type<float>   (copy_to, *it, copy_from, nearest_idx, result, *dst_it, float_names);
        handle_type<double>  (copy_to, *it, copy_from, nearest_idx, result, *dst_it, double_names);
        handle_type<int>     (copy_to, *it, copy_from, nearest_idx, result, *dst_it, int_names);
        handle_type<uint32_t>(copy_to, *it, copy_from, nearest_idx, result, *dst_it, uint_names);
        handle_type<uint8_t> (copy_to, *it, copy_from, nearest_idx, result, *dst_it, uchar_names);
        handle_type<Vector>  (copy_to, *it, copy_from, nearest_idx, result, *dst_it, vec_names);
    }

    return result;
}