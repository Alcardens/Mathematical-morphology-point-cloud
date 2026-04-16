#include "point_cloud_dilation_orientation.h"
#include "shaders.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>

static void check_gl(const char* tag) {
    GLenum e = glGetError();
    if (e != GL_NO_ERROR)
        throw std::runtime_error(std::string(tag) + " GL error: " + std::to_string(e));
}

GLuint PointCloudOrientationDilator::compile_compute(const char* src) {
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[4096]; glGetShaderInfoLog(sh, sizeof(buf), nullptr, buf);
        glDeleteShader(sh);
        throw std::runtime_error(std::string("Dilate compile:\n") + buf);
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, sh); glDeleteShader(sh);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[4096]; glGetProgramInfoLog(prog, sizeof(buf), nullptr, buf);
        glDeleteProgram(prog);
        throw std::runtime_error(std::string("Dilate link:\n") + buf);
    }
    return prog;
}

PointCloudOrientationDilator::PointCloudOrientationDilator(const OrientationDilationConfig& cfg)
    : m_cfg(cfg),
      m_hash(SpatialHashConfig{
          cfg.cell_size, cfg.min_dist, cfg.table_size, cfg.max_total_points
      })
{
    glGenBuffers(1, &m_ssbo_input);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_input);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)cfg.max_input_group * 4 * sizeof(float),
                 nullptr, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &m_ssbo_norm);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_norm);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)cfg.max_input_group * 4 * sizeof(float),
                 nullptr, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &m_ssbo_se);

    glGenBuffers(1, &m_ssbo_cand_pts);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_cand_pts);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)cfg.max_candidates * 4 * sizeof(float),
                 nullptr, GL_DYNAMIC_COPY);

    glGenBuffers(1, &m_ssbo_cand_cnt);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_cand_cnt);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(uint32_t), nullptr, GL_DYNAMIC_COPY);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    m_prog_dilate = compile_compute(SRC_DILATE_ORIENTATION);
}

PointCloudOrientationDilator::~PointCloudOrientationDilator() {
    glDeleteBuffers(1, &m_ssbo_input);
    glDeleteBuffers(1, &m_ssbo_norm);
    glDeleteBuffers(1, &m_ssbo_se);
    glDeleteBuffers(1, &m_ssbo_cand_pts);
    glDeleteBuffers(1, &m_ssbo_cand_cnt);
    glDeleteProgram(m_prog_dilate);
}

void PointCloudOrientationDilator::set_structuring_element(
        const std::vector<std::array<float, 3>>& se)
{
    m_se_count = (uint32_t)se.size();
    std::vector<float> buf;
    buf.reserve(m_se_count * 4);
    for (auto& p : se) {
        buf.push_back(p[0]); buf.push_back(p[1]);
        buf.push_back(p[2]); buf.push_back(0.f);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_se);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)buf.size() * sizeof(float), buf.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void PointCloudOrientationDilator::process_group(const std::vector<std::array<float, 3>>& group,
                                                 const std::vector<std::array<float, 3>>& normals) {
    if (group.empty() || m_se_count == 0) return;
    if (m_hash.is_full()) {
        std::cerr << "[Dilator] Output hash full — skipping group\n";
        return;
    }

    uint32_t n = (uint32_t)std::min(group.size(), (size_t)m_cfg.max_input_group);

    std::vector<float> buf;
    buf.reserve(n * 4);
    for (uint32_t i = 0; i < n; ++i) {
        buf.push_back(group[i][0]); buf.push_back(group[i][1]);
        buf.push_back(group[i][2]); buf.push_back(0.f);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_input);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    (GLsizeiptr)buf.size() * sizeof(float), buf.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    dispatch_dilation(n);

    std::vector<float> norm_buf;
    norm_buf.reserve(n * 4);
    for (uint32_t i = 0; i < n; ++i) {
        float nx = i < normals.size() ? normals[i][0] : 0.f;
        float ny = i < normals.size() ? normals[i][1] : 1.f;
        float nz = i < normals.size() ? normals[i][2] : 0.f;
        norm_buf.push_back(nx); norm_buf.push_back(ny);
        norm_buf.push_back(nz); norm_buf.push_back(0.f);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_norm);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    (GLsizeiptr)norm_buf.size() * sizeof(float), norm_buf.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    dispatch_dilation(n);

    // Read back accepted count, clamped to actually-written entries
    uint32_t accepted = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_cand_cnt);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &accepted);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    accepted = std::min(accepted, m_cfg.max_candidates);

    if (accepted > 0)
        m_hash.append_candidates(m_ssbo_cand_pts, accepted);
}

void PointCloudOrientationDilator::dispatch_dilation(uint32_t input_count) {
    uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_cand_cnt);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(m_prog_dilate);
    glUniform1ui(glGetUniformLocation(m_prog_dilate, "u_table_size"),  m_cfg.table_size);
    glUniform1f (glGetUniformLocation(m_prog_dilate, "u_cell_size"),   m_cfg.cell_size);
    glUniform1f (glGetUniformLocation(m_prog_dilate, "u_min_dist_sq"), m_cfg.min_dist * m_cfg.min_dist);
    glUniform1ui(glGetUniformLocation(m_prog_dilate, "u_num_input"),   input_count);
    glUniform1ui(glGetUniformLocation(m_prog_dilate, "u_se_count"),    m_se_count);
    glUniform1ui(glGetUniformLocation(m_prog_dilate, "u_max_cand"),    m_cfg.max_candidates);

    m_hash.bind(0, 1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_ssbo_input);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_ssbo_norm);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_ssbo_se);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_ssbo_cand_pts);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_ssbo_cand_cnt);

    uint32_t total_threads = input_count * m_se_count;
    glDispatchCompute((total_threads + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

std::vector<std::array<float, 3>> PointCloudOrientationDilator::get_result() const {
    return m_hash.download_points();
}

Point_set PointCloudOrientationDilator::get_result_cgal() const {
    auto pts = get_result();
    Point_set ps;
    for (auto& p : pts)
        ps.insert(Point(p[0], p[1], p[2]));
    return ps;
}
