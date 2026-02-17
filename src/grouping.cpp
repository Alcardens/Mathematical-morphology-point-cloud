#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <boost/property_map/property_map.hpp>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>

#include <CGAL/IO/read_ply_points.h>
#include <CGAL/IO/write_ply_points.h>
#include <CGAL/IO/Color.h>

#include "grouping.h"

typedef std::array<unsigned char, 4> Color; // RGBA
typedef std::tuple<Point, Color> PC;
typedef CGAL::Nth_of_tuple_property_map<0, PC> Point_map;
typedef CGAL::Nth_of_tuple_property_map<1, PC> Color_map;

namespace CGAL
{
    template<class F>
    struct Output_rep< ::Color, F > {
        const ::Color& c;
        static const bool is_specialized = true;
        Output_rep(const ::Color& c) : c(c) { }

        std::ostream& operator()(std::ostream& out) const
        {
            if (IO::is_ascii(out))
                out << int(c[0]) << " " << int(c[1]) << " " << int(c[2]) << " " << int(c[3]);
            else
                out.write(reinterpret_cast<const char*>(&c), sizeof(c));
            return out;
        }
    };
}

static Color palette_color(std::size_t i);

//int main()
//{
//    // Read input Point_set
//    Point_set data;
//    std::ifstream in("../input/Armadillo.ply", std::ios::binary);
//    if (!in) {
//        std::cerr << "Error: cannot open input PLY file\n";
//        return EXIT_FAILURE;
//    }
//    if (!CGAL::IO::read_PLY(in, data)) {
//        std::cerr << "Error: invalid or unsupported PLY file\n";
//        return EXIT_FAILURE;
//    }
//
//    // Split (use your SE diameter here; using 1.0 for now)
//    const double D = 1.5;
//    std::vector<Point_set> subsets = split_by_se_diameter(data, D);
//    std::cerr << "Created " << subsets.size() << " subsets\n";
//
//    // Build vector of (Point, Color) for output
//    std::vector<PC> out;
//    out.reserve(data.size());
//
//    for (std::size_t si = 0; si < subsets.size(); ++si) {
//        const Color col = palette_color(si);
//        std::cout << subsets[si].size() << "\n";
//
//        for (auto idx : subsets[si]) {
//            out.emplace_back(subsets[si].point(idx), col);
//        }
//    }
//
//    // Write one PLY with per-vertex colors (red, green, blue, alpha)
//    std::ofstream f("subsets_colored.ply", std::ios::binary);
//    if (!f) {
//        std::cerr << "Error: cannot open output file\n";
//        return EXIT_FAILURE;
//    }
//    CGAL::IO::set_binary_mode(f);
//
//    CGAL::IO::write_PLY_with_properties(
//        f, out,
//        CGAL::make_ply_point_writer(Point_map()),
//        std::make_tuple(Color_map(),
//                        CGAL::IO::PLY_property<unsigned char>("red"),
//                        CGAL::IO::PLY_property<unsigned char>("green"),
//                        CGAL::IO::PLY_property<unsigned char>("blue"),
//                        CGAL::IO::PLY_property<unsigned char>("alpha"))
//    );
//
//    std::cerr << "Wrote subsets_colored.ply\n";
//    return EXIT_SUCCESS;
//}


struct CellKey {
    int64_t x, y, z;
    bool operator==(const CellKey& o) const { return x==o.x && y==o.y && z==o.z; }
};

struct CellKeyHash {
    std::size_t operator()(const CellKey& k) const noexcept {
        // decent 64-bit mix
        auto h = [](int64_t v) -> std::size_t {
            uint64_t x = static_cast<uint64_t>(v);
            x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
            x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
            x ^= x >> 33;
            return static_cast<std::size_t>(x);
        };
        std::size_t hx = h(k.x), hy = h(k.y), hz = h(k.z);
        return hx ^ (hy + 0x9e3779b97f4a7c15ULL + (hx<<6) + (hx>>2))
                  ^ (hz + 0x9e3779b97f4a7c15ULL + (hy<<6) + (hy>>2));
    }
};

struct ColoredPoint {
    Point p;
    int color;
};

static CellKey cell_of(const Point& p, double cell_size)
{
    const double inv = 1.0 / cell_size;
    return CellKey{
        static_cast<int64_t>(std::floor(p.x() * inv)),
        static_cast<int64_t>(std::floor(p.y() * inv)),
        static_cast<int64_t>(std::floor(p.z() * inv))
    };
}

// Greedy grid-based coloring: points of same color are guaranteed > D apart.
std::vector<Point_set> split_by_se_diameter(const Point_set& data, double D)
{
    std::vector<Point_set> subsets;
    if (data.empty() || D <= 0.0) return subsets;

    const double D2 = D * D;
    const double cell_size = D;

    // grid maps cell -> list of (point,color)
    std::unordered_map<CellKey, std::vector<ColoredPoint>, CellKeyHash> grid;
    grid.reserve(data.size());

    std::vector<char> forbidden;

    for (auto idx : data) {
        const Point p = data.point(idx);
        const CellKey ck = cell_of(p, cell_size);

        int max_color_seen = -1;
        std::vector<int> neighbor_colors;
        neighbor_colors.reserve(64);

        // Check conflicts in 27 neighboring cells
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    CellKey nk{ck.x + dx, ck.y + dy, ck.z + dz};
                    auto it = grid.find(nk);
                    if (it == grid.end()) continue;

                    for (const auto& cp : it->second) {
                        if (CGAL::squared_distance(p, cp.p) <= D2) {
                            neighbor_colors.push_back(cp.color);
                            max_color_seen = std::max(max_color_seen, cp.color);
                        }
                    }
                }
            }
        }

        // Mark forbidden colors
        forbidden.assign(static_cast<std::size_t>(max_color_seen + 2), 0);
        for (int c : neighbor_colors)
            forbidden[static_cast<std::size_t>(c)] = 1;

        // Pick smallest available color
        int color = 0;
        while (color < static_cast<int>(forbidden.size()) &&
               forbidden[static_cast<std::size_t>(color)]) {
            ++color;
        }

        // Ensure enough Point_sets
        if (color >= static_cast<int>(subsets.size()))
            subsets.resize(static_cast<std::size_t>(color + 1));

        // Insert into that subset
        subsets[static_cast<std::size_t>(color)].insert(p);

        // Record into grid
        grid[ck].push_back(ColoredPoint{p, color});
    }

    return subsets;
}

static Color palette_color(std::size_t i)
{
    static const Color pal[30] = {
        {230,  25,  75, 255}, // red
        { 60, 180,  75, 255}, // green
        {255, 225,  25, 255}, // yellow
        {  0, 130, 200, 255}, // blue
        {245, 130,  48, 255}, // orange
        {145,  30, 180, 255}, // purple
        { 70, 240, 240, 255}, // cyan
        {240,  50, 230, 255}, // magenta
        {210, 245,  60, 255}, // lime
        {250, 190, 190, 255}, // pink

        {  0, 128, 128, 255}, // teal
        {230, 190, 255, 255}, // lavender
        {170, 110,  40, 255}, // brown
        {255, 250, 200, 255}, // beige
        {128,   0,   0, 255}, // maroon
        {170, 255, 195, 255}, // mint
        {128, 128,   0, 255}, // olive
        {255, 215, 180, 255}, // apricot
        {  0,   0, 128, 255}, // navy
        {128, 128, 128, 255}, // gray

        {255,  99,  71, 255}, // tomato
        {154, 205,  50, 255}, // yellowgreen
        {100, 149, 237, 255}, // cornflowerblue
        {218, 112, 214, 255}, // orchid
        { 64, 224, 208, 255}, // turquoise
        {255, 140,   0, 255}, // dark orange
        {138,  43, 226, 255}, // blue violet
        { 46, 139,  87, 255}, // sea green
        {255, 182, 193, 255}, // light pink
        {105, 105, 105, 255}  // dim gray
    };

    return pal[i % 30];
}



