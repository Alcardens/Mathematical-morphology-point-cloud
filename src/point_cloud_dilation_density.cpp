#include "point_cloud_dilation_density.h"
#include "shaders.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>

static void check_gl(const char* tag) {
    GLenum e = glGetError();
    if (e != GL_NO_ERROR)
        throw std::runtime_error(std::string(tag) + " GL error: " + std::to_string(e));
}

GLuint PointCloudDensityDilator::compile_compute(const char* src) {
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

PointCloudDensityDilator::PointCloudDensityDilator(const DensityDilationConfig& cfg)
    : m_cfg(cfg),
      m_hash(SpatialHashConfig{
          cfg.max_dens, cfg.max_dens, cfg.table_size, cfg.max_total_points
      })
{
    glGenBuffers(1, &m_ssbo_input);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_input);
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
    m_prog_dilate = compile_compute(SRC_DILATE_DENSITY);
}

PointCloudDensityDilator::~PointCloudDensityDilator() {
    glDeleteBuffers(1, &m_ssbo_input);
    glDeleteBuffers(1, &m_ssbo_se);
    glDeleteBuffers(1, &m_ssbo_cand_pts);
    glDeleteBuffers(1, &m_ssbo_cand_cnt);
    glDeleteProgram(m_prog_dilate);
}

void PointCloudDensityDilator::set_structuring_element(
        const std::vector<std::array<float, 4>>& se)
{
    m_se_count = (uint32_t)se.size();
    std::vector<float> buf;
    buf.reserve(m_se_count * 4);
    for (auto& p : se) {
        buf.push_back(p[0]); buf.push_back(p[1]);
        buf.push_back(p[2]); buf.push_back(p[3]);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_se);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)buf.size() * sizeof(float), buf.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void PointCloudDensityDilator::process_group(const std::vector<std::array<float, 4>>& group) {
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
        buf.push_back(group[i][2]); buf.push_back(group[i][3]);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_input);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    (GLsizeiptr)buf.size() * sizeof(float), buf.data());
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

void PointCloudDensityDilator::dispatch_dilation(uint32_t input_count) {
    uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_cand_cnt);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    glUseProgram(m_prog_dilate);
    glUniform1ui(glGetUniformLocation(m_prog_dilate, "u_table_size"),  m_cfg.table_size);
    glUniform1f (glGetUniformLocation(m_prog_dilate, "u_cell_size"),   m_cfg.min_dens);
    glUniform1f (glGetUniformLocation(m_prog_dilate, "u_min_dens"),    m_cfg.min_dens);
    glUniform1f (glGetUniformLocation(m_prog_dilate, "u_max_dens"),    m_cfg.max_dens);
    glUniform1f (glGetUniformLocation(m_prog_dilate, "u_increment"),   m_cfg.increment);
    glUniform1ui(glGetUniformLocation(m_prog_dilate, "u_num_input"),   input_count);
    glUniform1ui(glGetUniformLocation(m_prog_dilate, "u_se_count"),    m_se_count);
    glUniform1ui(glGetUniformLocation(m_prog_dilate, "u_max_cand"),    m_cfg.max_candidates);

    m_hash.bind(0, 1);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_ssbo_input);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_ssbo_se);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_ssbo_cand_pts);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_ssbo_cand_cnt);

    uint32_t total_threads = input_count * m_se_count;
    glDispatchCompute((total_threads + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

std::vector<std::array<float, 4>> PointCloudDensityDilator::get_result() const {
    const uint32_t n = m_hash.point_count();
    if (n == 0) return {};
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_hash.points_ssbo());
    const float* mapped = (const float*)glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0,
        (GLsizeiptr)n * 4 * sizeof(float), GL_MAP_READ_BIT);
    if (!mapped) throw std::runtime_error("glMapBufferRange failed");
    std::vector<std::array<float, 4>> out;
    out.reserve(n);
    for (uint32_t i = 0; i < n; ++i)
        out.push_back({ mapped[i*4], mapped[i*4+1], mapped[i*4+2], mapped[i*4+3] });
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return out;
}

Point_set PointCloudDensityDilator::get_result_cgal() const {
    auto pts = get_result();
    Point_set ps;
    auto score_map = ps.add_property_map<float>("density", 0.f).first;
    for (auto& p : pts) {
        auto it = ps.insert(Point(p[0], p[1], p[2]));
        score_map[*it] = p[3];
    }
    return ps;
}