#include "point_cloud_erosion.h"
#include "shaders.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>

static void check_gl(const char* tag) {
    GLenum e = glGetError();
    if (e != GL_NO_ERROR)
        throw std::runtime_error(std::string(tag) + " GL error: " + std::to_string(e));
}

GLuint PointCloudEroder::compile_compute(const char* src) {
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[4096]; glGetShaderInfoLog(sh, sizeof(buf), nullptr, buf);
        glDeleteShader(sh);
        throw std::runtime_error(std::string("Erode compile:\n") + buf);
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, sh); glDeleteShader(sh);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[4096]; glGetProgramInfoLog(prog, sizeof(buf), nullptr, buf);
        glDeleteProgram(prog);
        throw std::runtime_error(std::string("Erode link:\n") + buf);
    }
    return prog;
}

PointCloudEroder::PointCloudEroder(const ErosionConfig& cfg) : m_cfg(cfg) {
    if (cfg.table_size == 0 || (cfg.table_size & (cfg.table_size - 1)) != 0)
        throw std::invalid_argument("table_size must be a power of two");

    // Input hash table: open-addressing, 2 uints per slot
    uint64_t cell_bytes   = (uint64_t)cfg.table_size * 4 * sizeof(uint32_t);
    uint64_t input_bytes  = (uint64_t)cfg.max_input  * 4 * sizeof(float);
    uint64_t output_bytes = (uint64_t)cfg.max_output * 4 * sizeof(float);

    std::cout << "[PointCloudEroder] hash=" << cell_bytes  / (1024*1024) << " MB"
              << "  input=" << input_bytes  / (1024*1024) << " MB"
              << "  output=" << output_bytes / (1024*1024) << " MB\n";

    glGenBuffers(1, &m_ssbo_input_cells);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_input_cells);
    glBufferData(GL_SHADER_STORAGE_BUFFER, cell_bytes, nullptr, GL_DYNAMIC_COPY);
    check_gl("input cells alloc");

    glGenBuffers(1, &m_ssbo_input_pts);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_input_pts);
    glBufferData(GL_SHADER_STORAGE_BUFFER, input_bytes, nullptr, GL_DYNAMIC_DRAW);
    check_gl("input pts alloc");

    glGenBuffers(1, &m_ssbo_se);

    glGenBuffers(1, &m_ssbo_out_pts);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_pts);
    glBufferData(GL_SHADER_STORAGE_BUFFER, output_bytes, nullptr, GL_DYNAMIC_COPY);
    check_gl("output pts alloc");

    glGenBuffers(1, &m_ssbo_out_count);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_count);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(uint32_t), nullptr, GL_DYNAMIC_COPY);
    check_gl("output count alloc");

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    m_prog_insert = compile_compute(SRC_INSERT);
    m_prog_erode  = compile_compute(SRC_ERODE);
    m_prog_erode_score = compile_compute(SRC_ERODE_SCORE);
}

PointCloudEroder::~PointCloudEroder() {
    glDeleteBuffers(1, &m_ssbo_input_cells);
    glDeleteBuffers(1, &m_ssbo_input_pts);
    glDeleteBuffers(1, &m_ssbo_se);
    glDeleteBuffers(1, &m_ssbo_out_pts);
    glDeleteBuffers(1, &m_ssbo_out_count);
    glDeleteProgram(m_prog_insert);
    glDeleteProgram(m_prog_erode);
    glDeleteProgram(m_prog_erode_score);
}

void PointCloudEroder::set_structuring_element(
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

void PointCloudEroder::build_input_hash(uint32_t point_count) {
    // Clear the hash table to EMPTY_KEY (0xFFFFFFFF)
    uint32_t empty = 0x7FFFFFFFu;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_input_cells);
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI,
                      GL_RED_INTEGER, GL_UNSIGNED_INT, &empty);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // SRC_INSERT reads from OutputPts (binding 1) and writes to Cells (binding 0).
    // For erosion the "output" role is played by the input points, so we bind
    // m_ssbo_input_pts to binding 1.
    glUseProgram(m_prog_insert);
    glUniform1ui(glGetUniformLocation(m_prog_insert, "u_table_size"),   m_cfg.table_size);
    glUniform1f (glGetUniformLocation(m_prog_insert, "u_cell_size"),    m_cfg.cell_size);
    glUniform1ui(glGetUniformLocation(m_prog_insert, "u_num_points"),   point_count);
    glUniform1ui(glGetUniformLocation(m_prog_insert, "u_start_offset"), 0u);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssbo_input_cells);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_ssbo_input_pts);

    glDispatchCompute((point_count + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

uint32_t PointCloudEroder::erode(const std::vector<std::array<float, 3>>& input) {
    return dispatch_erode(input, m_prog_erode, /*write_all=*/false);
}

uint32_t PointCloudEroder::erode_score(const std::vector<std::array<float, 3>>& input) {
    return dispatch_erode(input, m_prog_erode_score, /*write_all=*/true);
}

// Shared implementation used by both erode() and erode_score().
uint32_t PointCloudEroder::dispatch_erode(
        const std::vector<std::array<float, 3>>& input,
        GLuint prog,
        bool write_all)
{
    if (input.empty() || m_se_count == 0) return 0;

    uint32_t n = (uint32_t)std::min(input.size(), (size_t)m_cfg.max_input);

    // ── 1. Upload all input points ─────────────────────────────────────────
    std::vector<float> buf;
    buf.reserve(n * 4);
    for (uint32_t i = 0; i < n; ++i) {
        buf.push_back(input[i][0]); buf.push_back(input[i][1]);
        buf.push_back(input[i][2]); buf.push_back(0.f);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_input_pts);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    (GLsizeiptr)buf.size() * sizeof(float), buf.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // ── 2. Build input hash ────────────────────────────────────────────────
    build_input_hash(n);

    // ── 3. Reset output counter ────────────────────────────────────────────
    uint32_t zero = 0;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_count);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &zero);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // ── 4. Dispatch ────────────────────────────────────────────────────────
    // Soft mode: u_max_output = n (all points written).
    // Hard mode: u_max_output = cfg.max_output (only survivors written).
    uint32_t max_out = write_all ? n : m_cfg.max_output;

    glUseProgram(prog);
    glUniform1ui(glGetUniformLocation(prog, "u_table_size"),  m_cfg.table_size);
    glUniform1f (glGetUniformLocation(prog, "u_cell_size"),   m_cfg.cell_size);
    glUniform1f (glGetUniformLocation(prog, "u_min_dist_sq"), m_cfg.min_dist * m_cfg.min_dist);
    glUniform1ui(glGetUniformLocation(prog, "u_num_input"),   n);
    glUniform1ui(glGetUniformLocation(prog, "u_se_count"),    m_se_count);
    glUniform1ui(glGetUniformLocation(prog, "u_max_output"),  max_out);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssbo_input_cells);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_ssbo_input_pts);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_ssbo_se);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_ssbo_out_pts);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_ssbo_out_count);

    glDispatchCompute((n + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // ── 5. Read back result count ──────────────────────────────────────────
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_count);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &m_result_count);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    m_result_count = std::min(m_result_count, max_out);

    std::cout << (write_all ? "[erode_score] " : "[erode] ")
              << n << " input → " << m_result_count << " output\n";
    return m_result_count;
}

std::vector<std::array<float, 3>> PointCloudEroder::get_result() const {
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

std::vector<std::array<float, 4>> PointCloudEroder::get_result_with_scores() const {
    if (m_result_count == 0) return {};
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_pts);
    const float* mapped = (const float*)glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0,
        (GLsizeiptr)m_result_count * 4 * sizeof(float), GL_MAP_READ_BIT);
    if (!mapped) throw std::runtime_error("glMapBufferRange failed");
    std::vector<std::array<float, 4>> out;
    out.reserve(m_result_count);
    for (uint32_t i = 0; i < m_result_count; ++i)
        out.push_back({ mapped[i*4], mapped[i*4+1], mapped[i*4+2], mapped[i*4+3] });
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return out;
}

Point_set PointCloudEroder::get_result_cgal() const {
    auto pts = get_result();
    Point_set ps;
    for (auto& p : pts)
        ps.insert(Point(p[0], p[1], p[2]));
    return ps;
}

Point_set PointCloudEroder::get_result_cgal_with_scores() const {
    auto pts = get_result_with_scores();
    Point_set ps;
    auto score_map = ps.add_property_map<float>("erosion_score", 0.f).first;
    for (auto& p : pts) {
        auto it = ps.insert(Point(p[0], p[1], p[2]));
        score_map[*it] = p[3];
    }
    return ps;
}