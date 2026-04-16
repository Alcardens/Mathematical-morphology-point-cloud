#include "grouping.h"

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3                                          Point;
typedef K::Vector_3                                         Vector;
typedef CGAL::Point_set_3<Point>                            Point_set;

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

// Flat representation of a point for grouping:
struct FlatPoint {
    float xyz[3];
    float prop;
    float normal[3];
};

static CellKey cell_of(const FlatPoint& p, double cell_size) {
    const double inv = 1.0 / cell_size;
    return CellKey{
        static_cast<int64_t>(std::floor(p.xyz[0] * inv)),
        static_cast<int64_t>(std::floor(p.xyz[1] * inv)),
        static_cast<int64_t>(std::floor(p.xyz[2] * inv))
    };
}

static std::vector<FlatPoint> to_flat(
    const Point_set&                  ps,
    bool                              include_normal,
    const std::optional<std::string>& property)
{
    std::vector<FlatPoint> out;
    out.reserve(ps.size());

    auto prop_opt = property
        ? ps.property_map<float>(*property)
        : std::optional<Point_set::Property_map<float>>{};

    // CGAL stores normals as a Vector_3 property named "normal"
    auto norm_opt = include_normal
        ? ps.property_map<Vector>("normal")
        : std::optional<Point_set::Property_map<Vector>>{};

    for (auto it = ps.begin(); it != ps.end(); ++it) {
        const Point& p = ps.point(*it);
        FlatPoint fp;
        fp.xyz[0] = (float)p.x();
        fp.xyz[1] = (float)p.y();
        fp.xyz[2] = (float)p.z();
        fp.prop   = prop_opt.has_value() ? (*prop_opt)[*it] : 0.f;
        if (norm_opt.has_value()) {
            const Vector& n = (*norm_opt)[*it];
            fp.normal[0] = (float)n.x();
            fp.normal[1] = (float)n.y();
            fp.normal[2] = (float)n.z();
        } else {
            fp.normal[0] = 0.f;
            fp.normal[1] = 1.f;
            fp.normal[2] = 0.f;
        }
        out.push_back(fp);
    }
    return out;
}

static Point_set from_flat(
    const std::vector<FlatPoint>&     pts,
    bool                              include_normal,
    const std::optional<std::string>& property)
{
    Point_set ps;

    Point_set::Property_map<float>  prop_map;
    Point_set::Property_map<Vector> norm_map;

    if (property)
        prop_map = ps.add_property_map<float>(*property, 0.f).first;
    if (include_normal)
        norm_map = ps.add_property_map<Vector>("normal", Vector(0, 1, 0)).first;

    for (const auto& fp : pts) {
        auto it = ps.insert(Point(fp.xyz[0], fp.xyz[1], fp.xyz[2]));
        if (property)
            prop_map[*it] = fp.prop;
        if (include_normal)
            norm_map[*it] = Vector(fp.normal[0], fp.normal[1], fp.normal[2]);
    }
    return ps;
}

std::vector<Point_set> split_by_se_diameter(
    const Point_set&                  data,
    double                            D,
    bool                              include_normal,
    const std::optional<std::string>& property)
{
    std::vector<Point_set> groups;
    if (data.empty() || D <= 0.0) return groups;

    const double D2        = D * D;
    const double cell_size = D;

    std::vector<FlatPoint> remaining = to_flat(data, include_normal, property);

    std::sort(remaining.begin(), remaining.end(),
              [](const FlatPoint& a, const FlatPoint& b) {
        if (a.xyz[0] != b.xyz[0]) return a.xyz[0] < b.xyz[0];
        if (a.xyz[1] != b.xyz[1]) return a.xyz[1] < b.xyz[1];
        return a.xyz[2] < b.xyz[2];
    });

    while (!remaining.empty()) {
        std::vector<FlatPoint> group_pts;
        std::vector<FlatPoint> next_remaining;
        next_remaining.reserve(remaining.size() / 2);

        std::unordered_map<CellKey, std::vector<FlatPoint>, CellKeyHash> grid;
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
                    double ddx = p.xyz[0]-gp.xyz[0];
                    double ddy = p.xyz[1]-gp.xyz[1];
                    double ddz = p.xyz[2]-gp.xyz[2];
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

        groups.push_back(from_flat(group_pts, include_normal, property));
        remaining = std::move(next_remaining);
    }

    std::sort(groups.begin(), groups.end(), [](const Point_set& a, const Point_set& b) {
        return a.size() > b.size();
    });

    return groups;
}
