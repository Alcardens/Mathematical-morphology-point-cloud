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
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);   // headless — no visible window
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

    Point_set input = multi_density_plane(1.0, 1.0, 1.0, 10);

    std::ofstream dout("../../Benchmarking/planes/plane_xz_10_d.ply", std::ios::binary);
    if (!dout) { std::cerr << "Error: cannot open output file\n"; return EXIT_FAILURE; }
    if (!CGAL::IO::write_PLY(dout, input)) { std::cerr << "Error: cannot write PLY\n"; return EXIT_FAILURE; }

    // Point_set se = regular_plane(1, 10);
    //
    // std::cout << "Loaded " << input.size() << " input points, "
    //           << se.size() << " SE points\n";
    //
    // Point_set output = erode_orientation_score_brute(input, se, 0.9, 90, true);
    //
    // std::cout << "Result " << output.size() << " output points";
    //
    // std::ofstream dout("../output/brute.ply", std::ios::binary);
    // if (!dout) { std::cerr << "Error: cannot open output file\n"; return EXIT_FAILURE; }
    // if (!CGAL::IO::write_PLY(dout, output)) { std::cerr << "Error: cannot write PLY\n"; return EXIT_FAILURE; }

    // ── Load input point cloud ──────────────────────────────────────────────
    // Point_set data;
    // std::ifstream in("../input/Armadillo.ply", std::ios::binary);
    // if (!in) { std::cerr << "Error: cannot open input PLY\n"; return EXIT_FAILURE; }
    // if (!CGAL::IO::read_PLY(in, data)) { std::cerr << "Error: invalid PLY\n"; return EXIT_FAILURE; }


    glfwDestroyWindow(win);
    glfwTerminate();
    return EXIT_SUCCESS;
}