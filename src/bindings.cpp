#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "erode.h"
#include "dilate.h"
#include "boolean.h"
#include "density.h"
#include "old_algorithm.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/IO/read_ply_points.h>
#include <CGAL/IO/write_ply_points.h>
#include <CGAL/Point_set_3/IO/LAS.h>

#include <GLFW/glfw3.h>
#include <GL/glew.h>

#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <filesystem>
#include <functional>

namespace py = pybind11;
namespace fs = std::filesystem;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3                                          Point;
typedef K::Vector_3                                         Vector;
typedef CGAL::Point_set_3<Point>                            Point_set;

// ── OpenGL context (created once at module load) ───────────────────────────
static GLFWwindow* g_window = nullptr;

static void init_gl_context()
{
    if (g_window) return;
    if (!glfwInit())
        throw std::runtime_error("glfwInit failed");
    glfwWindowHint(GLFW_VISIBLE,               GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,        GLFW_OPENGL_CORE_PROFILE);
    g_window = glfwCreateWindow(1, 1, "morphology_ctx", nullptr, nullptr);
    if (!g_window)
        throw std::runtime_error("glfwCreateWindow failed");
    glfwMakeContextCurrent(g_window);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
        throw std::runtime_error("glewInit failed");
}

// ── File I/O ───────────────────────────────────────────────────────────────

static Point_set load_point_set(const std::string& path)
{
    Point_set ps;
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".ply") {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open: " + path);
        if (!CGAL::IO::read_PLY(f, ps))
            throw std::runtime_error("Failed to read PLY: " + path);
    } else if (ext == ".las" || ext == ".laz") {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open: " + path);
        if (!CGAL::IO::read_LAS(f, ps))
            throw std::runtime_error("Failed to read LAS: " + path);
    } else {
        throw std::runtime_error("Unsupported file format: " + ext + " (use .ply or .las)");
    }

    return ps;
}

static void save_point_set(const std::string& path, const Point_set& ps)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open output: " + path);
    if (!CGAL::IO::write_PLY(f, ps))
        throw std::runtime_error("Failed to write PLY: " + path);
}

// ── Generic runners ────────────────────────────────────────────────────────
//
// run_operation: two input files (data + se), returns runtime in ms.
// run_single:    one input file, returns runtime in ms.
// run_pair:      two input files, no se semantics (for add/intersect/subtract).

static double run_operation(
    const std::string&                                           data_path,
    const std::string&                                           se_path,
    std::function<Point_set(const Point_set&, const Point_set&)> fn,
    const std::string&                                           output_path = "")
{
    Point_set data = load_point_set(data_path);
    Point_set se   = load_point_set(se_path);

    auto begin       = std::chrono::steady_clock::now();
    Point_set result = fn(data, se);
    auto end         = std::chrono::steady_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();

    if (!output_path.empty())
        save_point_set(output_path, result);

    return elapsed_ms;
}

// Single-input runner — for density_estimate which takes only one point cloud
static double run_single(
    const std::string&                    data_path,
    std::function<Point_set(const Point_set&)> fn,
    const std::string&                    output_path = "")
{
    Point_set data = load_point_set(data_path);

    auto begin       = std::chrono::steady_clock::now();
    Point_set result = fn(data);
    auto end         = std::chrono::steady_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();

    if (!output_path.empty())
        save_point_set(output_path, result);

    return elapsed_ms;
}

// ── Module ─────────────────────────────────────────────────────────────────

PYBIND11_MODULE(morphology, m)
{
    m.doc() = "GPU-accelerated 3D point cloud mathematical morphology";

    init_gl_context();

    // ── erode ────────────────────────────────────────────────────────────
    m.def("erode",
        [](const std::string& data, const std::string& se,
           float min_dist, const std::string& output) {
            return run_operation(data, se,
                [min_dist](const Point_set& d, const Point_set& s) {
                    return erode(d, s, min_dist); },
                output);
        },
        py::arg("data"), py::arg("se"), py::arg("min_dist"),
        py::arg("output") = "",
        "Hard erosion. Returns algorithm runtime in milliseconds.");

    m.def("erode_score",
        [](const std::string& data, const std::string& se,
           float min_dist, const std::string& output) {
            return run_operation(data, se,
                [min_dist](const Point_set& d, const Point_set& s) {
                    return erode_score(d, s, min_dist); },
                output);
        },
        py::arg("data"), py::arg("se"), py::arg("min_dist"),
        py::arg("output") = "",
        "Soft erosion. Returns algorithm runtime in milliseconds.");

    m.def("erode_density",
        [](const std::string& data, const std::string& se,
           float increment, const std::string& output) {
            return run_operation(data, se,
                [increment](const Point_set& d, const Point_set& s) {
                    return erode_density(d, s, increment); },
                output);
        },
        py::arg("data"), py::arg("se"), py::arg("increment"),
        py::arg("output") = "");

    m.def("erode_density_score",
        [](const std::string& data, const std::string& se,
           float increment, const std::string& output) {
            return run_operation(data, se,
                [increment](const Point_set& d, const Point_set& s) {
                    return erode_density_score(d, s, increment); },
                output);
        },
        py::arg("data"), py::arg("se"), py::arg("increment"),
        py::arg("output") = "");

    m.def("erode_orientation",
        [](const std::string& data, const std::string& se,
           float min_dist, const std::string& output) {
            return run_operation(data, se,
                [min_dist](const Point_set& d, const Point_set& s) {
                    return erode_orientation(d, s, min_dist); },
                output);
        },
        py::arg("data"), py::arg("se"), py::arg("min_dist"),
        py::arg("output") = "");

    m.def("erode_orientation_score",
        [](const std::string& data, const std::string& se,
           float min_dist, const std::string& output) {
            return run_operation(data, se,
                [min_dist](const Point_set& d, const Point_set& s) {
                    return erode_orientation_score(d, s, min_dist); },
                output);
        },
        py::arg("data"), py::arg("se"), py::arg("min_dist"),
        py::arg("output") = "");

    m.def("erode_orientation_brute",
        [](const std::string& data, const std::string& se,
           float min_dist, float angle, bool z_axis,
           const std::string& output) {
            return run_operation(data, se,
                [min_dist, angle, z_axis](const Point_set& d, const Point_set& s) {
                    return erode_orientation_brute(d, s, min_dist, angle, z_axis); },
                output);
        },
        py::arg("data"), py::arg("se"), py::arg("min_dist"),
        py::arg("angle") = 45.f, py::arg("z_axis") = false,
        py::arg("output") = "");

    m.def("erode_orientation_score_brute",
        [](const std::string& data, const std::string& se,
           float min_dist, float angle, bool z_axis,
           const std::string& output) {
            return run_operation(data, se,
                [min_dist, angle, z_axis](const Point_set& d, const Point_set& s) {
                    return erode_orientation_score_brute(d, s, min_dist, angle, z_axis); },
                output);
        },
        py::arg("data"), py::arg("se"), py::arg("min_dist"),
        py::arg("angle") = 45.f, py::arg("z_axis") = false,
        py::arg("output") = "");

    // ── dilate ───────────────────────────────────────────────────────────
    m.def("dilate",
        [](const std::string& data, const std::string& se,
           float min_dist, const std::string& output) {
            return run_operation(data, se,
                [min_dist](const Point_set& d, const Point_set& s) {
                    return dilate(d, s, min_dist); },
                output);
        },
        py::arg("data"), py::arg("se"), py::arg("min_dist"),
        py::arg("output") = "");

    m.def("dilate_density",
        [](const std::string& data, const std::string& se,
           float increment, const std::string& output) {
            return run_operation(data, se,
                [increment](const Point_set& d, const Point_set& s) {
                    return dilate_density(d, s, increment); },
                output);
        },
        py::arg("data"), py::arg("se"), py::arg("increment"),
        py::arg("output") = "");

    m.def("dilate_orientation",
        [](const std::string& data, const std::string& se,
           float min_dist, const std::string& output) {
            return run_operation(data, se,
                [min_dist](const Point_set& d, const Point_set& s) {
                    return dilate_orientation(d, s, min_dist); },
                output);
        },
        py::arg("data"), py::arg("se"), py::arg("min_dist"),
        py::arg("output") = "");

    // ── boolean ──────────────────────────────────────────────────────────
    m.def("add",
        [](const std::string& original, const std::string& addition,
           float min_dist, const std::string& output) {
            return run_operation(original, addition,
                [min_dist](const Point_set& a, const Point_set& b) {
                    return add(a, b, min_dist); },
                output);
        },
        py::arg("original"), py::arg("addition"), py::arg("min_dist"),
        py::arg("output") = "",
        "Add two point clouds, keeping only points from addition that are far enough from original.");

    m.def("intersect",
        [](const std::string& original, const std::string& intersection,
           float min_dist, const std::string& output) {
            return run_operation(original, intersection,
                [min_dist](const Point_set& a, const Point_set& b) {
                    return intersect(a, b, min_dist); },
                output);
        },
        py::arg("original"), py::arg("intersection"), py::arg("min_dist"),
        py::arg("output") = "",
        "Keep only points from original that have a neighbour in intersection.");

    m.def("subtract",
        [](const std::string& original, const std::string& subtraction,
           float min_dist, const std::string& output) {
            return run_operation(original, subtraction,
                [min_dist](const Point_set& a, const Point_set& b) {
                    return subtract(a, b, min_dist); },
                output);
        },
        py::arg("original"), py::arg("subtraction"), py::arg("min_dist"),
        py::arg("output") = "",
        "Keep only points from original that have no neighbour in subtraction.");

    m.def("copy_attributes",
        [](const std::string& copy_to, const std::string& copy_from,
           const std::string& output) {
            return run_operation(copy_to, copy_from,
                [](const Point_set& a, const Point_set& b) {
                    return copy_attributes(a, b); },
                output);
        },
        py::arg("copy_to"), py::arg("copy_from"),
        py::arg("output") = "",
        "Copy all property attributes from the nearest neighbour in copy_from to each point in copy_to.");

    // ── density ──────────────────────────────────────────────────────────
    m.def("density_estimate",
        [](const std::string& data, const std::string& output) {
            return run_single(data,
                [](const Point_set& d) { return density_estimate(d); },
                output);
        },
        py::arg("data"),
        py::arg("output") = "",
        "Estimate local point density for each point. Returns runtime in milliseconds.");

    // ── old (CPU) algorithms ──────────────────────────────────────────────
    m.def("erode_old",
        [](const std::string& data, const std::string& se,
           const std::string& output) {
            return run_operation(data, se,
                [](const Point_set& d, const Point_set& s) {
                    return erode_old(d, s); },
                output);
        },
        py::arg("data"), py::arg("se"),
        py::arg("output") = "",
        "CPU erosion (MATLAB port). Returns algorithm runtime in milliseconds.");

    m.def("dilate_old",
        [](const std::string& data, const std::string& se,
           const std::string& output) {
            return run_operation(data, se,
                [](const Point_set& d, const Point_set& s) {
                    return dilate_old(d, s); },
                output);
        },
        py::arg("data"), py::arg("se"),
        py::arg("output") = "",
        "CPU dilation (MATLAB port). Returns algorithm runtime in milliseconds.");
}

