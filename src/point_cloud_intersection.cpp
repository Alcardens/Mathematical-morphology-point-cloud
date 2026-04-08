#include "point_cloud_intersection.h"
#include "shaders.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>

static void check_gl(const char* tag) {
    GLenum e = glGetError();
    if (e != GL_NO_ERROR)
        throw std::runtime_error(std::string(tag) + " GL error: " + std::to_string(e));
}

GLuint PointCloudIntersector::compile_compute(const char* src) {
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

PointCloudIntersector::PointCloudIntersector(const IntersectionConfig& cfg)
    : m_cfg(cfg),
      m_hash(SpatialHashConfig{
          cfg.cell_size, cfg.min_dist, cfg.table_size, cfg.num_intersection
    })
{
    glGenBuffers(1, &m_ssbo_original_pts);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_original_pts);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)cfg.num_original * 4 * sizeof(float),
                 nullptr, GL_DYNAMIC_DRAW);
    check_gl("addition pts alloc");

    glGenBuffers(1, &m_ssbo_out_pts);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_pts);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)cfg.num_original * 4 * sizeof(float),
                 nullptr, GL_DYNAMIC_COPY);
    check_gl("output pts alloc");

    glGenBuffers(1, &m_ssbo_out_count);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_count);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(uint32_t), nullptr, GL_DYNAMIC_COPY);
    check_gl("output count alloc");

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    m_prog_intersect = compile_compute(SRC_INTERSECT);
    m_prog_subtract = compile_compute(SRC_SUBTRACT);
}

PointCloudIntersector::~PointCloudIntersector() {
    glDeleteBuffers(1, &m_ssbo_original_pts);
    glDeleteBuffers(1, &m_ssbo_out_pts);
    glDeleteBuffers(1, &m_ssbo_out_count);
    glDeleteProgram(m_prog_intersect);
    glDeleteProgram(m_prog_subtract);
}

uint32_t PointCloudIntersector::intersect(
        const std::vector<std::array<float, 3>>& original,
        const std::vector<std::array<float, 3>>& intersection)
{
    return dispatch_operation(original, intersection, m_prog_intersect, "intersect");
}

uint32_t PointCloudIntersector::subtract(
        const std::vector<std::array<float, 3>>& original,
        const std::vector<std::array<float, 3>>& subtract)
{
    return dispatch_operation(original, subtract, m_prog_subtract, "subtract");
}

uint32_t PointCloudIntersector::dispatch_operation(
        const std::vector<std::array<float, 3>>& original,
        const std::vector<std::array<float, 3>>& other,
        GLuint prog,
        const char* label)
{
    if (original.empty() || other.empty()) return 0;

    const uint32_t n_orig  = (uint32_t)std::min(original.size(), (size_t)m_cfg.num_original);
    const uint32_t n_other = (uint32_t)std::min(other.size(),    (size_t)m_cfg.num_intersection);

    // ── 1. Upload other points and build hash ──────────────────────────────
    m_hash.reset();
    m_hash.upload_points(other);

    // ── 2. Upload original points ──────────────────────────────────────────
    {
        std::vector<float> buf;
        buf.reserve(n_orig * 4);
        for (uint32_t i = 0; i < n_orig; ++i) {
            buf.push_back(original[i][0]); buf.push_back(original[i][1]);
            buf.push_back(original[i][2]); buf.push_back(0.f);
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_original_pts);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                        (GLsizeiptr)buf.size() * sizeof(float), buf.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    // ── 3. Reset output counter ────────────────────────────────────────────
    uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_count);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // ── 4. Dispatch ────────────────────────────────────────────────────────
    glUseProgram(prog);
    glUniform1ui(glGetUniformLocation(prog, "u_table_size"),   m_cfg.table_size);
    glUniform1f (glGetUniformLocation(prog, "u_cell_size"),    m_cfg.cell_size);
    glUniform1f (glGetUniformLocation(prog, "u_min_dist_sq"),  m_cfg.min_dist * m_cfg.min_dist);
    glUniform1ui(glGetUniformLocation(prog, "u_num_original"), n_orig);
    glUniform1ui(glGetUniformLocation(prog, "u_max_output"),   n_orig);

    m_hash.bind(0, 1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_ssbo_original_pts);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_ssbo_out_pts);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_ssbo_out_count);

    glDispatchCompute((n_orig + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // ── 5. Read back survivor count ────────────────────────────────────────
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_count);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &m_result_count);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    m_result_count = std::min(m_result_count, n_orig);

    std::cout << "[" << label << "] " << n_orig << " original, "
              << n_other << " other -> "
              << m_result_count << " surviving\n";
    return m_result_count;
}

std::vector<std::array<float, 3>> PointCloudIntersector::get_result() const {
    if (m_result_count == 0) return {};
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_pts);
    const float* mapped = (const float*)glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0,
        (GLsizeiptr)m_result_count * 4 * sizeof(float), GL_MAP_READ_BIT);
    if (!mapped) throw std::runtime_error("glMapBufferRange failed");
    std::vector<std::array<float, 3>> out;
    out.reserve(m_result_count);
    for (uint32_t i = 0; i < m_result_count; ++i)
        out.push_back({ mapped[i*4], mapped[i*4+1], mapped[i*4+2] });
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return out;
}

Point_set PointCloudIntersector::get_result_cgal() const {
    auto pts = get_result();
    Point_set ps;
    for (auto& p : pts)
        ps.insert(Point(p[0], p[1], p[2]));
    return ps;
}
