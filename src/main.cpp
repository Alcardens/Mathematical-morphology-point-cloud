#include "dilate.h"
#include "erode.h"
#include "density.h"
#include "boolean.h"
#include "geometry.h"
#include "old_algorithm.h"

#include <CGAL/IO/write_ply_points.h>
#include <CGAL/IO/read_ply_points.h>
#include <CGAL/Point_set_3/IO/LAS.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <chrono>

#include <fstream>
#include <iostream>
#include <cstdlib>

// Declared in your project's other translation units
Point_set fibonacci_sphere_multi_density(double distance0, double radius0);

// ── Create a hidden GLFW window solely for the OpenGL/compute context ──────
static GLFWwindow* create_gl_context() {
    if (!glfwInit()) {
        std::cerr << "Error: glfwInit failed\n";
        return nullptr;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* win = glfwCreateWindow(1, 1, "compute", nullptr, nullptr);
    if (!win) {
        std::cerr << "Error: glfwCreateWindow failed\n";
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(win);
    return win;
}

int main()
{
    // ── OpenGL context ──────────────────────────────────────────────────────
    GLFWwindow* win = create_gl_context();
    if (!win) return EXIT_FAILURE;

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Error: GLEW init failed\n";
        glfwTerminate();
        return EXIT_FAILURE;
    }

    Point_set input;

    std::ifstream in("../../Benchmarking/planes/density_plane_xy_01_1_50_200_4.ply", std::ios::binary);
    if (!in) { std::cerr << "Error: cannot open input PLY\n"; return EXIT_FAILURE; }
    if (!CGAL::IO::read_PLY(in, input)) { std::cerr << "Error: invalid PLY\n"; return EXIT_FAILURE; }

    // std::ofstream dout("../../Benchmarking/planes/cont_d_plane_xy_01_50.ply", std::ios::binary);
    // if (!dout) { std::cerr << "Error: cannot open output file\n"; return EXIT_FAILURE; }
    // if (!CGAL::IO::write_PLY(dout, input)) { std::cerr << "Error: cannot write PLY\n"; return EXIT_FAILURE; }

    // // ── Export lines (n: 1-10) ────────────────────────────────────────────────
    // for (int n = 0; n <= 9; ++n)
    // {
    //     double min_dist = 0.55 - double(n) * 0.05;
    //     double max_dist = min_dist + n * 0.1;
    //     Point_set se = multi_density_plane(min_dist, max_dist, 0.1, 5);
    //
    //     std::string path = "../../Benchmarking/planes/multi_d_plane_xy_01_1_" + std::to_string(n + 1) + "_5.ply";
    //     std::ofstream sout(path, std::ios::binary);
    //     if (!sout) { std::cerr << "Error: cannot open " << path << "\n"; return EXIT_FAILURE; }
    //     if (!CGAL::IO::write_PLY(sout, se)) { std::cerr << "Error: cannot write " << path << "\n"; return EXIT_FAILURE; }
    //     std::cout << "Wrote " << path << " (" << se.size() << " points)\n";
    // }

    // // ── Export planes (n: 2-10) ───────────────────────────────────────────────
    // for (int n = 2; n <= 10; ++n)
    // {
    //     Point_set se2 = regular_plane(10.0 / (n - 1), n);
    //
    //     std::string path = "../../Benchmarking/planes/plane_xy_" + std::to_string(n) + ".ply";
    //     std::ofstream s2out(path, std::ios::binary);
    //     if (!s2out) { std::cerr << "Error: cannot open " << path << "\n"; return EXIT_FAILURE; }
    //     if (!CGAL::IO::write_PLY(s2out, se2)) { std::cerr << "Error: cannot write " << path << "\n"; return EXIT_FAILURE; }
    //     std::cout << "Wrote " << path << " (" << se2.size() << " points)\n";
    // }

    Point_set se;

    std::ifstream sin("../../Benchmarking/planes/cont_d_plane_xy_01_50.ply", std::ios::binary);
    if (!sin) { std::cerr << "Error: cannot open input PLY\n"; return EXIT_FAILURE; }
    if (!CGAL::IO::read_PLY(sin, se)) { std::cerr << "Error: invalid PLY\n"; return EXIT_FAILURE; }

    std::cout << "Loaded " << input.size() << " input points, "
              << se.size() << " SE points\n";

    Point_set output = dilate_density(input, se, 0.1);

    std::cout << "Result " << output.size() << " output points";

    std::ofstream dout("../output/dens_dil.ply", std::ios::binary);
    if (!dout) { std::cerr << "Error: cannot open output file\n"; return EXIT_FAILURE; }
    if (!CGAL::IO::write_PLY(dout, output)) { std::cerr << "Error: cannot write PLY\n"; return EXIT_FAILURE; }

    // ── Load input point cloud ──────────────────────────────────────────────
    // Point_set data;
    // std::ifstream in("../input/Armadillo.ply", std::ios::binary);
    // if (!in) { std::cerr << "Error: cannot open input PLY\n"; return EXIT_FAILURE; }
    // if (!CGAL::IO::read_PLY(in, data)) { std::cerr << "Error: invalid PLY\n"; return EXIT_FAILURE; }


    glfwDestroyWindow(win);
    glfwTerminate();
    return EXIT_SUCCESS;
}