#pragma once

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>

#include <chrono>
#include <vector>
#include <cmath>
#include <iostream>

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef K::Point_3                                           Point;
typedef CGAL::Point_set_3<Point>                             Point_set;
typedef CGAL::Search_traits_3<K>                             Traits;
typedef CGAL::Kd_tree<Traits>                                Tree;
typedef CGAL::Orthogonal_k_neighbor_search<Traits>           KNN;

// Convert Point_set to vector of Points
static std::vector<Point> to_vec(const Point_set& ps)
{
    std::vector<Point> out;
    out.reserve(ps.size());
    for (auto it = ps.begin(); it != ps.end(); ++it)
        out.push_back(ps.point(*it));
    return out;
}

static double estimate_d_d(const std::vector<Point>& data_pts,
                           const Tree&               data_tree,
                           uint32_t                  sample_size = 500)
{
    const uint32_t n = (uint32_t)data_pts.size();
    sample_size = std::min(sample_size, n);

    // Draw a random sample of indices without replacement
    std::vector<uint32_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0u);
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(sample_size);

    double sum = 0.0;
    int    cnt = 0;
    for (uint32_t idx : indices) {
        KNN search(data_tree, data_pts[idx], 6);  // k=6: self + 5 neighbours
        int i = 0;
        for (auto it = search.begin(); it != search.end(); ++it, ++i) {
            if (i == 0) continue;  // skip self
            sum += std::sqrt(CGAL::to_double(it->second));
            ++cnt;
        }
    }
    return cnt > 0 ? sum / cnt : 1.0;
}


// ── dilate_old ────────────────────────────────────────────────────────────
Point_set dilate_old(const Point_set& data, const Point_set& se)
{
    if (data.empty() || se.empty()) return data;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    std::vector<Point> data_pts = to_vec(data);
    std::vector<Point> se_pts   = to_vec(se);

    Tree data_tree(data_pts.begin(), data_pts.end());
    double d_d = estimate_d_d(data_pts, data_tree);
    double d_t = d_d / 1.5;

    std::cout << "d_t: " << d_t << std::endl;

    Point_set data_n_dil;

    // Scroll points
    for (const Point& p : data_pts) {
        double tx = CGAL::to_double(p.x()) - CGAL::to_double(se_pts[0].x());
        double ty = CGAL::to_double(p.y()) - CGAL::to_double(se_pts[0].y());
        double tz = CGAL::to_double(p.z()) - CGAL::to_double(se_pts[0].z());

        std::vector<Point> se_to_data;
        se_to_data.reserve(se_pts.size() - 1);
        for (size_t j = 1; j < se_pts.size(); ++j) {
            se_to_data.emplace_back(
                CGAL::to_double(se_pts[j].x()) + tx,
                CGAL::to_double(se_pts[j].y()) + ty,
                CGAL::to_double(se_pts[j].z()) + tz
            );
        }

        for (const Point& c : se_to_data) {
            KNN search(data_tree, c, 1);
            double dist = std::sqrt(CGAL::to_double(search.begin()->second));
            if (dist > d_t)
                data_n_dil.insert(c);
        }
    }

    // Filter new SE repeated points (optimization)
    Point_set pc_n_dil;
    for (auto it = data_n_dil.begin(); it != data_n_dil.end(); ++it) {
        const Point& c = data_n_dil.point(*it);
        if (pc_n_dil.empty()) {
            pc_n_dil.insert(c);
            continue;
        }
        std::vector<Point> accepted = to_vec(pc_n_dil);
        Tree accepted_tree(accepted.begin(), accepted.end());
        KNN search(accepted_tree, c, 1);
        double dist = std::sqrt(CGAL::to_double(search.begin()->second));
        if (dist > d_t)
            pc_n_dil.insert(c);
    }

    // data_dilated = [data(:,1:3);pc_n_dil.Location];
    Point_set data_dilated;
    for (const Point& p : data_pts)
        data_dilated.insert(p);
    for (auto it = pc_n_dil.begin(); it != pc_n_dil.end(); ++it)
        data_dilated.insert(pc_n_dil.point(*it));

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    return data_dilated;
}

// ── erode_old ─────────────────────────────────────────────────────────────
Point_set erode_old(const Point_set& data, const Point_set& se)
{
    if (data.empty() || se.empty()) return data;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    std::vector<Point> data_pts = to_vec(data);
    std::vector<Point> se_pts   = to_vec(se);

    // Estimate average distance between points
    Tree data_tree(data_pts.begin(), data_pts.end());
    double d_d = estimate_d_d(data_pts, data_tree);
    double d_t = d_d / 1.5;

    Point_set data_n_er;

    for (const Point& p : data_pts) {
        double tx = CGAL::to_double(p.x()) - CGAL::to_double(se_pts[0].x());
        double ty = CGAL::to_double(p.y()) - CGAL::to_double(se_pts[0].y());
        double tz = CGAL::to_double(p.z()) - CGAL::to_double(se_pts[0].z());

        std::vector<Point> se_data;
        se_data.reserve(se_pts.size() - 1);
        for (size_t j = 1; j < se_pts.size(); ++j) {
            se_data.emplace_back(
                CGAL::to_double(se_pts[j].x()) + tx,
                CGAL::to_double(se_pts[j].y()) + ty,
                CGAL::to_double(se_pts[j].z()) + tz
            );
        }

        bool all_matched = true;
        for (const Point& s : se_data) {
            KNN search(data_tree, s, 1);
            double dist = std::sqrt(CGAL::to_double(search.begin()->second));
            if (dist > d_t) {
                all_matched = false;
                break;
            }
        }
        if (all_matched)
            data_n_er.insert(p);
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    return data_n_er;
}
