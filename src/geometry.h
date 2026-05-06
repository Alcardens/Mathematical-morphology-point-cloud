#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/IO/write_ply_points.h>
#include <CGAL/IO/read_ply_points.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef K::Point_3                                           Point;
typedef CGAL::Point_set_3<Point>                             Point_set;

// One shell fibonacci

Point_set fibonacci_sphere(        double distance,
                                   double radius)
{
    Point_set points;

    if (distance == 0 || radius <= 0.0) return points;
    int samples = 4 * radius * radius / (distance * distance);

    const double phi = CGAL_PI * (std::sqrt(5.0) - 1.0);

    if (samples == 1) {
        points.insert(Point(0.0, radius, 0.0));
        return points;
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
        points.insert(Point(radius * x, radius * y, radius * z));
    }

    return points;
}

static void fibonacci_sphere_shell(Point_set& out,
                                   double distance,
                                   double radius)
{
    if (distance == 0 || radius <= 0.0) return;
    int samples = 4 * radius * radius / (distance * distance);

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

// Multi-shell fibonacci
Point_set fibonacci_sphere_multi_density(double distance0 = 1.0,
                                         double radius0 = 1.0)
{
    Point_set points;

    if (distance0 == 0 || radius0 <= 0.0) return points;

    for (double R = radius0; R > 0.0; R -= distance0) {
        fibonacci_sphere_shell(points, distance0, R);
    }

    return points;
}

Point_set hollow_cube(double distance = 1.0,
                      int n = 10000)
{
    Point_set points;

    if (distance == 0.0 || n <= 0) return points;

    double edge = floor(2 + sqrt(6 * (n - 2)) / 6);

    double limit_1 = - (edge - 1) / 2 * distance;
    double limit_2 = (edge - 1) / 2 * distance;

    // left and right plane
    for (int j = 0; j < edge; j++)
        for (int k = 0; k < edge; k++)
        {
            double y = (j - (edge - 1) / 2) * distance;
            double z = (k - (edge - 1) / 2) * distance;
            points.insert(Point(limit_1 + rand() * distance * 0.1 / double(RAND_MAX),
                y + rand() * distance * 0.1 / double(RAND_MAX),
                z + rand() * distance * 0.1 / double(RAND_MAX)));
            points.insert(Point(limit_2 + rand() * distance * 0.1 / double(RAND_MAX),
                y + rand() * distance * 0.1 / double(RAND_MAX),
                z + rand() * distance * 0.1 / double(RAND_MAX)));
        }

    // top and bottom plane
    for (int i = 1; i < edge - 1; i++)
        for (int j = 0; j < edge; j++)
        {
            double x = (i - (edge - 1) / 2) * distance;
            double y = (j - (edge - 1) / 2) * distance;
            points.insert(Point(x + rand() * distance * 0.1 / double(RAND_MAX),
                y + rand() * distance * 0.1 / double(RAND_MAX),
                limit_1 + rand() * distance * 0.1 / double(RAND_MAX)));
            points.insert(Point(x + rand() * distance * 0.1 / double(RAND_MAX),
                y + rand() * distance * 0.1 / double(RAND_MAX),
                limit_2 + rand() * distance * 0.1 / double(RAND_MAX)));
        }

    // front and back plane
    for (int i = 1; i < edge - 1; i++)
        for (int k = 1; k < edge - 1; k++)
        {
            double x = (i - (edge - 1) / 2) * distance;
            double z = (k - (edge - 1) / 2) * distance;
            points.insert(Point(x + rand() * distance * 0.1 / double(RAND_MAX),
                limit_1 + rand() * distance * 0.1 / double(RAND_MAX),
                z + rand() * distance * 0.1 / double(RAND_MAX)));
            points.insert(Point(x + rand() * distance * 0.1 / double(RAND_MAX),
                limit_2 + rand() * distance * 0.1 / double(RAND_MAX),
                z + rand() * distance * 0.1 / double(RAND_MAX)));
        }

    return points;
}

Point_set regular_line(double distance = 1.0,
                      int n = 5, string axis = "x")
{
    Point_set points;

    if (distance == 0.0 || n <= 0) return points;

    double start = - (n - 1) / 2 * distance;
    for (int i = 0; i < n; i++)
    {
        double x = start + i * distance;
        if (axis == "x")
        {
            points.insert(Point(x, 0.0, 0.0));
        } else if (axis == "y")
        {
            points.insert(Point(0.0, x, 0.0));
        } else
        {
            points.insert(Point(0.0, 0.0, x));
        }
    }

    return points;
}

Point_set regular_plane(double distance = 1.0,
                      int edge = 5)
{
    Point_set points;

    if (distance == 0.0 || edge <= 0) return points;

    double start = - (edge - 1) / 2 * distance;
    for (int i = 0; i < edge; i++)
        for (int j = 0; j < edge; j++)
        {
            double x = start + i * distance;
            double z = start + j * distance;
            points.insert(Point(x, z, 0));
        }

    return points;
}

Point_set plane_density(double min_dist = 0.1, double max_dist = 1.0,
                        int side = 20, int steps = 20, double power = 3.0)
{
    Point_set points;
    if (min_dist <= 0.0 || max_dist <= 0.0 || side <= 0 || steps <= 0)
        return points;

    float x = 0;
    for (int i = 0; i < steps; i++)
    {
        // t goes 0→1 linearly, then curved to spend more time near 0
        double t    = std::pow((double)i / (steps - 1), power);
        float  dist = static_cast<float>(min_dist + (max_dist - min_dist) * t);

        float y = 0;
        for (int j = 0; j < (int)(side / dist); j++)
        {
            y += dist;
            points.insert(Point(x, y, 0));
        }
        x += dist;
    }

    return points;
}

Point_set multi_density_sphere(double radius = 5, double min_dist = 0.1, double max_dist = 1.0, double increment = 0.1)
{
    Point_set points;
    auto density_map = points.add_property_map<float>("density", 0.f).first;

    for (double density = min_dist;
         density <= max_dist + increment * 0.5;  // 0.5 * increment tolerance for float rounding
         density += increment)
    {
        Point_set shell = fibonacci_sphere_multi_density(density, radius);

        for (auto it = shell.begin(); it != shell.end(); ++it) {
            const Point& p  = shell.point(*it);
            auto dst_it     = points.insert(p);
            density_map[*dst_it] = static_cast<float>(density);
        }
    }

    return points;
}

Point_set multi_density_plane(double min_dist = 0.1, double max_dist = 1.0, double increment = 0.1, int edge = 5)
{
    Point_set points;
    auto density_map = points.add_property_map<float>("density", 0.f).first;

    for (double density = min_dist;
         density <= max_dist + increment * 0.5;  // 0.5 * increment tolerance for float rounding
         density += increment)
    {
        int edge_points = edge / density;
        Point_set plane = regular_plane(density, edge_points);

        for (auto it = plane.begin(); it != plane.end(); ++it) {
            const Point& p  = plane.point(*it);
            auto dst_it     = points.insert(p);
            density_map[*dst_it] = static_cast<float>(density);
        }
    }

    return points;
}

Point_set arrow_z(
    double shaft_width  = 0.3,
    double shaft_height = 1.0,
    double head_width   = 0.7,
    double head_height  = 0.8,
    int    shaft_side_n = 8,
    int    head_base_n  = 7,
    int    head_side_n  = 8)
{
    Point_set points;

    const double tip_z = shaft_height + head_height;

    // ── Shaft: left and right vertical edges ──────────────────────────────
    for (int i = 0; i <= shaft_side_n; ++i) {
        double z = (shaft_height / shaft_side_n) * i;
        points.insert(Point(-shaft_width, 0, z));
        if (i > 0)
            points.insert(Point( shaft_width, 0, z));
    }

    // Shaft base centre
    points.insert(Point(0, 0, 0));

    // ── Arrowhead base: horizontal edge at z = shaft_height ───────────────
    for (int i = 0; i <= head_base_n; ++i) {
        double x = -head_width + (2.0 * head_width / head_base_n) * i;
        points.insert(Point(x, 0, shaft_height));
    }

    // ── Arrowhead: left and right slanted edges converging to tip ─────────
    for (int i = 1; i <= head_side_n; ++i) {
        double t   = (double)i / head_side_n;
        double z   = shaft_height + head_height * t;
        double x_l = -head_width * (1.0 - t);
        double x_r =  head_width * (1.0 - t);
        points.insert(Point(x_l, 0, z));
        points.insert(Point(x_r, 0, z));
    }

    // Tip
    points.insert(Point(0, 0, tip_z));

    return points;
}

Point_set se_downsample(const Point_set& se)
{
    if (se.empty()) return Point_set{};

    std::vector<Point> pts;
    pts.reserve(se.size());
    for (auto it = se.begin(); it != se.end(); ++it)
        pts.push_back(se.point(*it));

    const size_t n = pts.size();
    size_t first_idx = 0;
    double best_dist_sq = std::numeric_limits<double>::max();
    for (size_t i = 0; i < n; ++i) {
        double d_sq = CGAL::to_double(CGAL::squared_distance(pts[i], Point(0, 0, 0)));
        if (d_sq < best_dist_sq) {
            best_dist_sq = d_sq;
            first_idx    = i;
        }
    }

    std::vector<double> dist_to_selected(n, std::numeric_limits<double>::max());

    std::vector<size_t> order;        // selected indices in order
    std::vector<double> densities;    // density label per selected point
    std::vector<bool>   selected(n, false);

    order.reserve(n);
    densities.reserve(n);

    order.push_back(first_idx);
    selected[first_idx] = true;
    densities.push_back(0.0);

    for (size_t i = 0; i < n; ++i) {
        if (selected[i]) continue;
        double d = std::sqrt(CGAL::to_double(
            CGAL::squared_distance(pts[i], pts[first_idx])));
        dist_to_selected[i] = d;
    }

    while (order.size() < n) {
        size_t best_idx = 0;
        double best_d   = -1.0;
        for (size_t i = 0; i < n; ++i) {
            if (!selected[i] && dist_to_selected[i] > best_d) {
                best_d   = dist_to_selected[i];
                best_idx = i;
            }
        }

        double density = dist_to_selected[best_idx];

        if (order.size() == 1)
            densities[0] = density;

        order.push_back(best_idx);
        densities.push_back(density);
        selected[best_idx] = true;

        for (size_t i = 0; i < n; ++i) {
            if (selected[i]) continue;
            double d = std::sqrt(CGAL::to_double(
                CGAL::squared_distance(pts[i], pts[best_idx])));
            if (d < dist_to_selected[i])
                dist_to_selected[i] = d;
        }
    }

    Point_set result;
    auto density_map = result.add_property_map<float>("density", 0.f).first;

    for (size_t k = 0; k < order.size(); ++k) {
        auto it = result.insert(pts[order[k]]);
        density_map[*it] = static_cast<float>(densities[k]);
    }

    std::cout << "[se_downsample] " << n << " input points"
              << "  density range=["
              << *std::min_element(densities.begin(), densities.end()) << ", "
              << *std::max_element(densities.begin(), densities.end()) << "]\n";

    return result;
}



