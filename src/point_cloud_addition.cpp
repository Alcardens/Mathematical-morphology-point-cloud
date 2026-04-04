#include "point_cloud_addition.h"
#include "shaders.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>

static void check_gl(const char* tag) {
    GLenum e = glGetError();
    if (e != GL_NO_ERROR)
        throw std::runtime_error(std::string(tag) + " GL error: " + std::to_string(e));
}

GLuint PointCloudAdder::compile_compute(const char* src) {
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[4096]; glGetShaderInfoLog(sh, sizeof(buf), nullptr, buf);
        glDeleteShader(sh);
        throw std::runtime_error(std::string("Add compile:\n") + buf);
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, sh); glDeleteShader(sh);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[4096]; glGetProgramInfoLog(prog, sizeof(buf), nullptr, buf);
        glDeleteProgram(prog);
        throw std::runtime_error(std::string("Add link:\n") + buf);
    }
    return prog;
}

PointCloudAdder::PointCloudAdder(const AdditionConfig& cfg)
    : m_cfg(cfg),
      m_hash(SpatialHashConfig{
          cfg.cell_size, cfg.min_dist, cfg.table_size, cfg.num_addition + cfg.num_original
    })
{
    glGenBuffers(1, &m_ssbo_addition_pts);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_addition_pts);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)cfg.num_addition * 4 * sizeof(float),
                 nullptr, GL_DYNAMIC_DRAW);
    check_gl("addition pts alloc");

    glGenBuffers(1, &m_ssbo_out_pts);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_pts);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 ((GLsizeiptr)cfg.num_original + (GLsizeiptr)cfg.num_addition) * 4 * sizeof(float),
                 nullptr, GL_DYNAMIC_COPY);
    check_gl("output pts alloc");

    glGenBuffers(1, &m_ssbo_out_count);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_count);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(uint32_t), nullptr, GL_DYNAMIC_COPY);
    check_gl("output count alloc");

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    m_prog_add = compile_compute(SRC_ADD);
}

PointCloudAdder::~PointCloudAdder() {
    glDeleteBuffers(1, &m_ssbo_addition_pts);
    glDeleteBuffers(1, &m_ssbo_out_pts);
    glDeleteBuffers(1, &m_ssbo_out_count);
    glDeleteProgram(m_prog_add);
}

uint32_t PointCloudAdder::add(const std::vector<std::array<float, 3>>& original,
    const std::vector<std::array<float, 3>>& addition)
{
    if (original.empty()) return 0;

    const uint32_t n_original = (uint32_t)std::min(original.size(), (size_t)m_cfg.num_original);
    const uint32_t n_addition = (uint32_t)std::min(addition.size(), (size_t)m_cfg.num_addition);


    // ── 1. Upload originals directly into the hash ────────────────────────
    m_hash.reset();
    m_hash.upload_points(original);

    // ── 2. Upload addition points ──────────────────────────────────────────
    if (n_addition > 0) {
        std::vector<float> buf;
        buf.reserve(n_addition * 4);
        for (uint32_t i = 0; i < n_addition; ++i) {
            buf.push_back(addition[i][0]); buf.push_back(addition[i][1]);
            buf.push_back(addition[i][2]); buf.push_back(0.f);
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_addition_pts);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                        (GLsizeiptr)buf.size() * sizeof(float), buf.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    // ── 3. Reset output counter ────────────────────────────────────────────
    uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_count);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // ── 4. Dispatch ADD ───────────────────────────────────────────────────
    glUseProgram(m_prog_add);
    glUniform1ui(glGetUniformLocation(m_prog_add, "u_table_size"),   m_cfg.table_size);
    glUniform1f (glGetUniformLocation(m_prog_add, "u_cell_size"),    m_cfg.cell_size);
    glUniform1f (glGetUniformLocation(m_prog_add, "u_min_dist_sq"),  m_cfg.min_dist * m_cfg.min_dist);
    glUniform1ui(glGetUniformLocation(m_prog_add, "u_num_addition"), n_addition);
    glUniform1ui(glGetUniformLocation(m_prog_add, "u_max_output"),   n_addition + n_original);

    // binding 0 = hash cells, binding 1 = original pts (in hash)
    m_hash.bind(0, 1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_ssbo_addition_pts);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_ssbo_out_pts);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_ssbo_out_count);

    glDispatchCompute((n_addition + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // ── 5. Read back accepted count ────────────────────────────────────────
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_count);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &m_result_count);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    m_result_count = std::min(m_result_count, m_cfg.num_addition);

    // ── 6. Insert accepted additions into hash ────────────────────────────
    if (m_result_count > 0)
        m_hash.append_candidates(m_ssbo_out_pts, m_result_count);

    std::cout << "[add] " << n_original << " original + "
              << m_result_count << " accepted additions"
              << " = " << (n_original + m_result_count) << " total\n";
    return n_original + m_result_count;
}

std::vector<std::array<float, 3>> PointCloudAdder::get_result() const {
    return m_hash.download_points();
}

Point_set PointCloudAdder::get_result_cgal() const {
    auto pts = get_result();
    Point_set ps;
    for (auto& p : pts)
        ps.insert(Point(p[0], p[1], p[2]));
    return ps;
}