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

struct CellKey {
    int64_t x, y, z;
    bool operator==(const CellKey& o) const { return x==o.x && y==o.y && z==o.z; }
};

struct CellKeyHash {
    std::size_t operator()(const CellKey& k) const noexcept {
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

static CellKey cell_of(const std::array<float,4>& p, double cell_size) {
    const double inv = 1.0 / cell_size;
    return CellKey{
        static_cast<int64_t>(std::floor(p[0] * inv)),
        static_cast<int64_t>(std::floor(p[1] * inv)),
        static_cast<int64_t>(std::floor(p[2] * inv))
    };
}

static std::vector<std::array<float,4>>
to_vec4(const Point_set& ps, const std::optional<std::string>& property)
{
    std::vector<std::array<float,4>> out;
    out.reserve(ps.size());

    auto prop_map_opt = property
        ? ps.property_map<float>(*property)
        : std::optional<Point_set::Property_map<float>>{};

    for (auto it = ps.begin(); it != ps.end(); ++it) {
        const Point& p = ps.point(*it);
        float w = prop_map_opt.has_value() ? (*prop_map_opt)[*it] : 0.f;
        out.push_back({ (float)p.x(), (float)p.y(), (float)p.z(), w });
    }
    return out;
}

static Point_set from_vec4(const std::vector<std::array<float,4>>& pts,
                           const std::optional<std::string>& property)
{
    Point_set ps;
    Point_set::Property_map<float> prop_map;
    if (property)
        prop_map = ps.add_property_map<float>(*property, 0.f).first;

    for (const auto& p : pts) {
        auto it = ps.insert(Point(p[0], p[1], p[2]));
        if (property)
            prop_map[*it] = p[3];
    }
    return ps;
}

std::vector<Point_set> split_by_se_diameter(
    const Point_set&                  data,
    double                            D,
    const std::optional<std::string>& property)
{
    std::vector<Point_set> groups;
    if (data.empty() || D <= 0.0) return groups;

    const double D2        = D * D;
    const double cell_size = D;

    // Convert to flat array — property value rides in w, 0 if absent
    std::vector<std::array<float,4>> remaining = to_vec4(data, property);

    // Spatially sort on xyz for better cache locality
    std::sort(remaining.begin(), remaining.end(),
              [](const std::array<float,4>& a, const std::array<float,4>& b) {
        if (a[0] != b[0]) return a[0] < b[0];
        if (a[1] != b[1]) return a[1] < b[1];
        return a[2] < b[2];
    });

    while (!remaining.empty()) {
        std::vector<std::array<float,4>> group_pts;
        std::vector<std::array<float,4>> next_remaining;
        next_remaining.reserve(remaining.size() / 2);

        std::unordered_map<CellKey, std::vector<std::array<float,4>>, CellKeyHash> grid;
        grid.reserve(remaining.size() / 8);

        for (const auto& p : remaining) {
            const CellKey ck = cell_of(p, cell_size);
            bool conflicts   = false;

            for (int dx = -1; dx <= 1 && !conflicts; ++dx)
            for (int dy = -1; dy <= 1 && !conflicts; ++dy)
            for (int dz = -1; dz <= 1 && !conflicts; ++dz) {
                auto it = grid.find({ ck.x+dx, ck.y+dy, ck.z+dz });
                if (it == grid.end()) continue;
                for (const auto& gp : it->second) {
                    double ddx = p[0]-gp[0], ddy = p[1]-gp[1], ddz = p[2]-gp[2];
                    if (ddx*ddx + ddy*ddy + ddz*ddz <= D2) { conflicts = true; break; }
                }
            }

            if (!conflicts) {
                group_pts.push_back(p);
                grid[ck].push_back(p);
            } else {
                next_remaining.push_back(p);
            }
        }

        groups.push_back(from_vec4(group_pts, property));
        remaining = std::move(next_remaining);
    }

    std::sort(groups.begin(), groups.end(), [](const Point_set& a, const Point_set& b) {
        return a.size() > b.size();
    });

    return groups;
}