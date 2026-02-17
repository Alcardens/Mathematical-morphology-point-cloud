#ifndef GROUPING_H
#define GROUPING_H

#include <vector>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
typedef K::Point_3                                           Point;
typedef CGAL::Point_set_3<Point>                             Point_set;

std::vector<Point_set> split_by_se_diameter(const Point_set& data, double D);

#endif //GROUPING_H
