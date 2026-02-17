#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <vector>
#include <CGAL/IO/write_ply_points.h>
#include <CGAL/IO/read_ply_points.h>
#include <CGAL/Kd_tree.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Vector_3.h>
#include "grouping.h"

typedef CGAL::Search_traits_3<K>                             Traits;
typedef CGAL::Kd_tree<Traits>                                Tree;
typedef CGAL::Orthogonal_k_neighbor_search<Traits>           KNN;

double average_spacing = 0.000150;
Point_set fibonacci_sphere_multi_density(std::size_t samples, double radius);
Point_set dilate(const Point_set& data, const Point_set& se, double threshold_scale, std::size_t progress_every);

int main()
{
    // Create structuring element
    Point_set se = fibonacci_sphere_multi_density(50, 1.5);

    // Export structuring element as .ply
    std::ofstream out("fibonacci_sphere.ply", std::ios::binary);
    if (!out) {
        std::cerr << "Error: cannot open output file\n";
        return EXIT_FAILURE;
    }
    if (!CGAL::IO::write_PLY(out, se)) {
        std::cerr << "Error: cannot write PLY file\n";
        return EXIT_FAILURE;
    }

    // Load input point cloud as Point_set
    Point_set data;
    std::ifstream in("../input/Armadillo.ply", std::ios::binary);
    if (!in) {
        std::cerr << "Error: cannot open input PLY file\n";
        return EXIT_FAILURE;
    }
    if (!CGAL::IO::read_PLY(in, data)) {
        std::cerr << "Error: invalid or unsupported PLY file\n";
        return EXIT_FAILURE;
    }

    // Apply the dilation
    Point_set dilation = dilate(data, se, 1.0, 10);

    // Export the dilation as .ply
    std::ofstream dout("../output/dilation.ply", std::ios::binary);
    if (!dout) {
        std::cerr << "Error: cannot open output file\n";
        return EXIT_FAILURE;
    }
    if (!CGAL::IO::write_PLY(dout, dilation)) {
        std::cerr << "Error: cannot write PLY file\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// Progress bar
static void print_progress(std::size_t current,
                           std::size_t total,
                           std::size_t bar_width = 50)
{
    const double progress = static_cast<double>(current) / static_cast<double>(total);
    const std::size_t pos = static_cast<std::size_t>(bar_width * progress);

    std::cout << "\r[";
    for (std::size_t i = 0; i < bar_width; ++i) {
        if (i < pos)       std::cout << "=";
        else if (i == pos) std::cout << ">";
        else               std::cout << " ";
    }

    std::cout << "] "
              << std::setw(3) << static_cast<int>(progress * 100.0) << "% ("
              << current << "/" << total << ")"
              << std::flush;
}

// Estimate the average distance between points in the input data set
static double estimate_average_knn_distance(const std::vector<Point>& pts,
                                           std::size_t k_neighbors = 5,
                                           std::size_t max_samples = 500)
{
    if (pts.size() < 2) return 0.0;

    const std::size_t k = std::min<std::size_t>(k_neighbors, pts.size() - 1);
    if (k == 0) return 0.0;

    Tree tree(pts.begin(), pts.end());
    tree.build();

    const std::size_t n = pts.size();
    const std::size_t step = std::max<std::size_t>(1, n / std::min(n, max_samples));

    double sum = 0.0;
    std::size_t count = 0;

    for (std::size_t i = 0; i < n; i += step) {
        // Ask for k+1 neighbors: [itself, then k nearest others]
        KNN search(tree, pts[i], static_cast<unsigned int>(k + 1));

        auto it = search.begin();
        if (it == search.end()) continue;

        // Skip itself
        ++it;

        std::size_t taken = 0;
        for (; it != search.end() && taken < k; ++it, ++taken) {
            const double d2 = it->second;   // squared distance
            sum += std::sqrt(d2);
            ++count;
        }
    }

    std::cout << sum / static_cast<double>(count);

    return (count > 0) ? (sum / static_cast<double>(count)) : 0.0;
}

Point_set dilate(const Point_set& data,
                 const Point_set& se,
                 double threshold_scale = 1.0,
                 std::size_t progress_every = 10)
{
    Point_set out;

    // Seed output with data points
    std::vector<Point> out_pts;
    out_pts.reserve(data.size() + data.size() * se.size()); // rough upper bound

    for (auto idx : data) {
        out_pts.push_back(data.point(idx));
    }

    // Edge cases
    if (out_pts.empty() || se.empty()) {
        out.reserve(out_pts.size());
        for (const auto& p : out_pts) out.insert(p);
        return out;
    }

    // Convert SE into offsets once
    std::vector<K::Vector_3> offsets;
    offsets.reserve(se.size());
    for (auto sidx : se) {
        const Point s = se.point(sidx);
        offsets.emplace_back(s.x(), s.y(), s.z());
    }

    // Estimate spacing and threshold
    const double avg_knn    = estimate_average_knn_distance(out_pts, 5);
    const double threshold  = threshold_scale * avg_knn;
    const double threshold2 = threshold * threshold;

    // TODO: Calculate SE diameter for grouping
    double se_diameter = 3.0;

    // Split data into groups based on SE diameter
    std::vector<Point_set> groups = split_by_se_diameter(data, se_diameter);

    const std::size_t total_groups = groups.size();
    std::size_t processed_groups = 0;

    // Process each group
    for (const auto& group : groups) {
        // Build KD-tree on current output points
        Tree tree(out_pts.begin(), out_pts.end());
        tree.build();

        // Collect new points from this group in parallel
        std::vector<Point> group_new_points;
        std::mutex group_mutex;

        // Convert group to vector for parallel processing
        std::vector<Point> group_points;
        group_points.reserve(group.size());
        for (auto gidx : group) {
            group_points.push_back(group.point(gidx));
        }

        // Process points in group in parallel
        #pragma omp parallel
        {
            std::vector<Point> thread_local_points;

            #pragma omp for schedule(dynamic)
            for (std::size_t i = 0; i < group_points.size(); ++i) {
                const Point& p = group_points[i];

                for (const auto& off : offsets) {
                    const Point c = p + off;

                    // Search in existing tree
                    KNN search(tree, c, 1);
                    const auto it = search.begin();
                    if (it != search.end() && it->second > threshold2) {
                        thread_local_points.push_back(c);
                    }
                }
            }

            // Merge thread-local results
            if (!thread_local_points.empty()) {
                std::lock_guard<std::mutex> lock(group_mutex);
                group_new_points.insert(group_new_points.end(),
                                       thread_local_points.begin(),
                                       thread_local_points.end());
            }
        }

        // Add all new points from this group to output
        out_pts.insert(out_pts.end(), group_new_points.begin(), group_new_points.end());

        ++processed_groups;

        // Progress update per group
        if ((processed_groups % progress_every) == 0 || processed_groups == total_groups) {
            print_progress(processed_groups, total_groups);
        }
    }

    std::cout << "\n";

    // Build Point_set result
    out.reserve(out_pts.size());
    for (const auto& pnt : out_pts) out.insert(pnt);

    return out;
}

// Generate ONE shell (single radius) with N samples
static void fibonacci_sphere_shell(Point_set& out,
                                   std::size_t samples,
                                   double radius)
{
    if (samples == 0 || radius <= 0.0) return;

    const double phi = CGAL_PI * (std::sqrt(5.0) - 1.0);

    if (samples == 1) {
        out.insert(Point(0.0, radius, 0.0));
        return;
    }

    for (std::size_t i = 0; i < samples; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(samples - 1);

        // Unit sphere
        const double y = 1.0 - t * 2.0; // [-1, 1]
        const double r = std::sqrt(std::max(0.0, 1.0 - y * y));
        const double theta = phi * static_cast<double>(i);

        const double x = std::cos(theta) * r;
        const double z = std::sin(theta) * r;

        // Scale to desired radius
        out.insert(Point(radius * x, radius * y, radius * z));
    }
}

// Generate multiple shells with constant surface-area-per-point density
Point_set fibonacci_sphere_multi_density(std::size_t samples0 = 1000,
                                        double radius0 = 1.0)
{
    Point_set points;

    if (samples0 == 0 || radius0 <= 0.0) return points;

    // density0 = surface area per point (length^2 per point)
    const double density0  = (4.0 * CGAL_PI * radius0 * radius0) / static_cast<double>(samples0);

    // distance ~ characteristic spacing on the surface (length)
    const double distance0 = std::sqrt(density0);

    // Optional: reserve a rough upper bound (not exact)
    points.reserve(samples0 * 2);

    // Iterate shells: R shrinks by ~distance0 each time, N shrinks to keep density0 constant
    for (double R = radius0; R > 0.0; R -= distance0) {
        // Keep constant surface-area-per-point: N(R) = area / density0
        const double area = 4.0 * CGAL_PI * R * R;
        std::size_t N = static_cast<std::size_t>(std::llround(area / density0));

        if (N == 0) break;

        fibonacci_sphere_shell(points, N, R);
    }

    return points;
}
