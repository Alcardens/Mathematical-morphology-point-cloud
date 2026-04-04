#ifndef GROUPING_H
#define GROUPING_H

#include <vector>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef K::Point_3                                           Point;
typedef CGAL::Point_set_3<Point>                             Point_set;

std::vector<Point_set> split_by_se_diameter(
    const Point_set&           data,
    double                     D,
    const std::optional<std::string>& property = std::nullopt);

#endif //GROUPING_H
