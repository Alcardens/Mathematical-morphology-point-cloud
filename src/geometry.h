#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/IO/write_ply_points.h>
#include <CGAL/IO/read_ply_points.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef K::Point_3                                           Point;
typedef CGAL::Point_set_3<Point>                             Point_set;

// One shell fibonacci
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
                      int n = 5)
{
    Point_set points;

    if (distance == 0.0 || n <= 0) return points;

    double start = - (n - 1) / 2 * distance;
    for (int i = 0; i < n; i++)
    {
        double x = start + i * distance;
        points.insert(Point(x, 0.0, 0.0));
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

Point_set plane_density(double min_dist = 0.1, double max_dist = 1.0, int side = 20, int steps = 20)
{
    Point_set points;

    if (min_dist <= 0.0 || max_dist <= 0.0 || side <= 0.0 || steps <= 0) return points;

    float dist = min_dist;
    float increment = (max_dist - min_dist) / steps;
    float x = 0;
    float y = 0;

    for (int i = 0; i < steps; i++)
    {
        y = 0;
        for (int j = 0; j < side / dist; j++)
        {
            y += dist;
            points.insert(Point(x, y, 0));
        }
        x += dist;
        dist += increment;
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
