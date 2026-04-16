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

    // ── Create structuring element ──────────────────────────────────────────
    // Point_set se = fibonacci_sphere_multi_density(1.0, 2.0);
    // Point_set se = regular_line(0.02, 5);
    // Point_set se = regular_plane(1.0, 8);
    // Point_set se = multi_density_sphere(3.0, 0.8, 1.2, 0.1);

    // std::ofstream out("structuring element.ply", std::ios::binary);
    // if (!out) { std::cerr << "Error: cannot open output file\n"; return EXIT_FAILURE; }
    // if (!CGAL::IO::write_PLY(out, se)) { std::cerr << "Error: cannot write PLY\n"; return EXIT_FAILURE; }

    // ── Create hollow cube ──────────────────────────────────────────────────
    // Point_set cube = hollow_cube(1.0, 800);

    // std::ofstream bout("hollow_cube.ply", std::ios::binary);
    // if (!bout) { std::cerr << "Error: cannot open output file\n"; return EXIT_FAILURE; }
    // if (!CGAL::IO::write_PLY(bout, cube)) { std::cerr << "Error: cannot write PLY\n"; return EXIT_FAILURE; }

    // ── Create density plane  ─────────────────────────────────────────────────
    Point_set input;
    std::ifstream in("../output/intersection.ply", std::ios::binary);
    if (!in) { std::cerr << "Error: cannot open input PLY\n"; return EXIT_FAILURE; }
    if (!CGAL::IO::read_PLY(in, input)) { std::cerr << "Error: invalid PLY\n"; return EXIT_FAILURE; }

    Point_set se;
    std::ifstream is("../input/vasca_left.ply", std::ios::binary);
    if (!is) { std::cerr << "Error: cannot open input PLY\n"; return EXIT_FAILURE; }
    if (!CGAL::IO::read_PLY(is, se)) { std::cerr << "Error: invalid PLY\n"; return EXIT_FAILURE; }

    std::cout << "Loaded " << input.size() << " input points, "
              << se.size() << " SE points\n";

    Point_set output = copy_attributes(input, se);

    std::cout << "Result " << output.size() << " output points";

    std::ofstream dout("../output/copy_att.ply", std::ios::binary);
    if (!dout) { std::cerr << "Error: cannot open output file\n"; return EXIT_FAILURE; }
    if (!CGAL::IO::write_PLY(dout, output)) { std::cerr << "Error: cannot write PLY\n"; return EXIT_FAILURE; }

    // std::ofstream bout("density_plane.ply", std::ios::binary);
    // if (!bout) { std::cerr << "Error: cannot open output file\n"; return EXIT_FAILURE; }
    // if (!CGAL::IO::write_PLY(bout, density_plane)) { std::cerr << "Error: cannot write PLY\n"; return EXIT_FAILURE; }

    // ── Load input point cloud ──────────────────────────────────────────────
    // Point_set data;
    // std::ifstream in("../input/Armadillo.ply", std::ios::binary);
    // if (!in) { std::cerr << "Error: cannot open input PLY\n"; return EXIT_FAILURE; }
    // if (!CGAL::IO::read_PLY(in, data)) { std::cerr << "Error: invalid PLY\n"; return EXIT_FAILURE; }

    // Point_set se;
    // std::ifstream sin ("../output/vasca_dilation.ply", std::ios::binary);
    // if (!sin) { std::cerr << "Error: cannot open input PLY\n"; return EXIT_FAILURE; }
    // if (!CGAL::IO::read_PLY(sin, se)) { std::cerr << "Error: invalid PLY\n"; return EXIT_FAILURE

    // Point_set data;
    // std::ifstream in ("../input/vasca_left.las", std::ios::binary);
    // if (!in) { std::cerr << "Error: cannot open input LAS\n"; return EXIT_FAILURE; }
    // if (!CGAL::IO::read_LAS(in, data)) { std::cerr << "Error: invalid LAS\n"; return EXIT_FAILURE; }
    //
    // std::cout << "Loaded " << data.size() << " input points, "
    //           << se.size() << " SE points\n";

    // ── Dilate ─────────────────────────────────────────────────────────────
    // for (float i = 6; i <= 12; i++)
    // {
    //     Point_set cube = fibonacci_sphere_multi_density(1.0,  i/2);
    //     std::cout << "Input: " << cube.size() << " points\n";
    //
    //     Point_set se = fibonacci_sphere_multi_density(1.0, 1.0);
    //
    //     Point_set erosion = dilate(cube, se, 1.02);
    //     std::cout << "Output: " << erosion.size() << " points\n";
    // }

    // Point_set cube = fibonacci_sphere_multi_density(1.0,  6.0);
    // Point_set se = fibonacci_sphere_multi_density(1.0, 1.0);
    // Point_set dilation = dilate_old(cube, se);

    // ── Erode ──────────────────────────────────────────────────────────────
    // Point_set erosion = dilate(data, se, 0.15);
    // std::cout << "Output: " << erosion.size() << " points\n";

    // ── Add ────────────────────────────────────────────────────────────────
    // Point_set addition = add(cube, se, 0.8);
    // std::cout << "Output: " << addition.size() << " points\n";

    // ── Add ────────────────────────────────────────────────────────────────
    // Point_set intersection = subtract(data, se, 0.03);
    // std::cout << "Output: " << intersection.size() << " points\n";

    // ── Density ────────────────────────────────────────────────────────────
    // Point_set density = density_estimate(data);
    // std::cout << "Output: " << density.size() << " points\n";

    // ── Dilation_density ───────────────────────────────────────────────────
    // Point_set dilation_density = dilate_density(data, se, 0.05);
    // std::cout << "Output: " << dilation_density.size() << " points\n";

    // ── Export ─────────────────────────────────────────────────────────────
    // std::ofstream iout("../input/input.ply", std::ios::binary);
    // if (!iout) { std::cerr << "Error: cannot open output file\n"; return EXIT_FAILURE; }
    // if (!CGAL::IO::write_PLY(iout, cube)) { std::cerr << "Error: cannot write PLY\n"; return EXIT_FAILURE; }
    //
    // std::ofstream dout("../output/dilation_old.ply", std::ios::binary);
    // if (!dout) { std::cerr << "Error: cannot open output file\n"; return EXIT_FAILURE; }
    // if (!CGAL::IO::write_PLY(dout, dilation)) { std::cerr << "Error: cannot write PLY\n"; return EXIT_FAILURE; }

    // ── Export ─────────────────────────────────────────────────────────────
    // std::ofstream dout("../output/vasca_dilation.ply", std::ios::binary);
    // if (!dout) { std::cerr << "Error: cannot open output file\n"; return EXIT_FAILURE; }
    // if (!CGAL::IO::write_PLY(dout, erosion)) { std::cerr << "Error: cannot write PLY\n"; return EXIT_FAILURE; }

    // ── Export ─────────────────────────────────────────────────────────────
    // std::ofstream dout("../output/addition.ply", std::ios::binary);
    // if (!dout) { std::cerr << "Error: cannot open output file\n"; return EXIT_FAILURE; }
    // if (!CGAL::IO::write_PLY(dout, addition)) { std::cerr << "Error: cannot write PLY\n"; return EXIT_FAILURE; }

    // ── Export ─────────────────────────────────────────────────────────────
    // std::ofstream dout("../output/density.ply", std::ios::binary);
    // if (!dout) { std::cerr << "Error: cannot open output file\n"; return EXIT_FAILURE; }
    // if (!CGAL::IO::write_PLY(dout, density)) { std::cerr << "Error: cannot write PLY\n"; return EXIT_FAILURE; }

    // ── Export ─────────────────────────────────────────────────────────────
    // std::ofstream dout("../output/subtraction.ply", std::ios::binary);
    // if (!dout) { std::cerr << "Error: cannot open output file\n"; return EXIT_FAILURE; }
    // if (!CGAL::IO::write_PLY(dout, intersection)) { std::cerr << "Error: cannot write PLY\n"; return EXIT_FAILURE; }

    glfwDestroyWindow(win);
    glfwTerminate();
    return EXIT_SUCCESS;
}