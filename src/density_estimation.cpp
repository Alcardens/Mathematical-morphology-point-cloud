#include "density_estimation.h"
#include "shaders.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>

static void check_gl(const char* tag) {
    GLenum e = glGetError();
    if (e != GL_NO_ERROR)
        throw std::runtime_error(std::string(tag) + " GL error: " + std::to_string(e));
}

GLuint DensityEstimator::compile_compute(const char* src) {
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[4096]; glGetShaderInfoLog(sh, sizeof(buf), nullptr, buf);
        glDeleteShader(sh);
        throw std::runtime_error(std::string("Density compile:\n") + buf);
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, sh); glDeleteShader(sh);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[4096]; glGetProgramInfoLog(prog, sizeof(buf), nullptr, buf);
        glDeleteProgram(prog);
        throw std::runtime_error(std::string("Density link:\n") + buf);
    }
    return prog;
}

DensityEstimator::DensityEstimator(const DensityConfig& cfg)
    : m_cfg(cfg),
      m_hash(SpatialHashConfig{
          cfg.cell_size, cfg.cell_size, cfg.table_size, cfg.num_points
      })
{
    glGenBuffers(1, &m_ssbo_out_dist);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_dist);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)cfg.num_points * sizeof(float),
                 nullptr, GL_DYNAMIC_COPY);
    check_gl("out dist alloc");
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    m_prog_density = compile_compute(SRC_DEN);
}

DensityEstimator::~DensityEstimator() {
    glDeleteBuffers(1, &m_ssbo_out_dist);
    glDeleteProgram(m_prog_density);
}

void DensityEstimator::density_estimation(const std::vector<std::array<float, 3>>& points)
{
    if (points.empty()) return;
    const uint32_t n = (uint32_t)std::min(points.size(), (size_t)m_cfg.num_points);

    m_hash.reset();
    m_hash.upload_points(points);

    glUseProgram(m_prog_density);

    glUniform1ui(glGetUniformLocation(m_prog_density, "u_table_size"), m_cfg.table_size);
    glUniform1f (glGetUniformLocation(m_prog_density, "u_cell_size"),  m_cfg.cell_size);
    glUniform1ui(glGetUniformLocation(m_prog_density, "u_num_points"), n);
    glUniform1i (glGetUniformLocation(m_prog_density, "u_max_shell"),  (int)m_cfg.max_shell);

    m_hash.bind(0, 1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_ssbo_out_dist);

    glDispatchCompute((n + 255) / 256, 1, 1);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

std::vector<std::array<float, 4>> DensityEstimator::get_result() const {
    const uint32_t n = m_hash.point_count();
    if (n == 0) return {};

    // Read xyz from the hash's point buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_hash.points_ssbo());
    const float* xyz = (const float*)glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0,
        (GLsizeiptr)n * 4 * sizeof(float), GL_MAP_READ_BIT);
    if (!xyz) throw std::runtime_error("glMapBufferRange (xyz) failed");

    // Read density values from the separate output buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_dist);
    const float* dist = (const float*)glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0,
        (GLsizeiptr)n * sizeof(float), GL_MAP_READ_BIT);
    if (!dist) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_hash.points_ssbo());
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        throw std::runtime_error("glMapBufferRange (dist) failed");
    }

    std::vector<std::array<float, 4>> out;
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i)
        out.push_back({ xyz[i*4], xyz[i*4+1], xyz[i*4+2], dist[i] });

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);  // unmap dist
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_hash.points_ssbo());
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);  // unmap xyz
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return out;
}

Point_set DensityEstimator::get_result_cgal() const {
    auto pts = get_result();
    Point_set ps;
    auto score_map = ps.add_property_map<float>("density", 0.f).first;
    for (auto& p : pts) {
        auto it = ps.insert(Point(p[0], p[1], p[2]));
        score_map[*it] = p[3];
    }
    return ps;
}