#include "point_cloud_erosion_orientation_brute.h"
#include "shaders.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>

static void check_gl(const char* tag) {
    GLenum e = glGetError();
    if (e != GL_NO_ERROR)
        throw std::runtime_error(std::string(tag) + " GL error: " + std::to_string(e));
}

GLuint PointCloudBruteEroder::compile_compute(const char* src) {
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

PointCloudBruteEroder::PointCloudBruteEroder(const BruteErosionConfig& cfg) : m_cfg(cfg) {
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

    glGenBuffers(1, &m_ssbo_se_ld);
    glGenBuffers(1, &m_ssbo_se_hd);

    glGenBuffers(1, &m_ssbo_out_pts);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_pts);
    glBufferData(GL_SHADER_STORAGE_BUFFER, output_bytes, nullptr, GL_DYNAMIC_COPY);
    check_gl("output pts alloc");

    glGenBuffers(1, &m_ssbo_out_norm);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_norm);
    glBufferData(GL_SHADER_STORAGE_BUFFER, output_bytes, nullptr, GL_DYNAMIC_COPY);
    check_gl("output norm alloc");

    glGenBuffers(1, &m_ssbo_out_count);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_count);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(uint32_t), nullptr, GL_DYNAMIC_COPY);
    check_gl("output count alloc");

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    m_prog_insert       = compile_compute(SRC_INSERT);
    m_prog_erode        = compile_compute(SRC_ERODE_ORIENTATION_BRUTE);
    m_prog_erode_score  = compile_compute(SRC_ERODE_ORIENTATION_BRUTE_SCORE);
}

PointCloudBruteEroder::~PointCloudBruteEroder() {
    glDeleteBuffers(1, &m_ssbo_input_cells);
    glDeleteBuffers(1, &m_ssbo_input_pts);
    glDeleteBuffers(1, &m_ssbo_se_ld);
    glDeleteBuffers(1, &m_ssbo_se_hd);
    glDeleteBuffers(1, &m_ssbo_out_pts);
    glDeleteBuffers(1, &m_ssbo_out_norm);
    glDeleteBuffers(1, &m_ssbo_out_count);
    glDeleteProgram(m_prog_insert);
    glDeleteProgram(m_prog_erode);
    glDeleteProgram(m_prog_erode_score);
}

void PointCloudBruteEroder::set_structuring_element(
        const std::vector<std::array<float, 3>>& se_ld, const std::vector<std::array<float, 3>>& se_hd)
{
    m_se_ld_count = (uint32_t)se_ld.size();
    std::vector<float> buf_ld;
    buf_ld.reserve(m_se_ld_count * 4);
    for (auto& p : se_ld) {
        buf_ld.push_back(p[0]); buf_ld.push_back(p[1]);
        buf_ld.push_back(p[2]); buf_ld.push_back(0.f);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_se_ld);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)buf_ld.size() * sizeof(float), buf_ld.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    m_se_hd_count = (uint32_t)se_hd.size();
    std::vector<float> buf_hd;
    buf_hd.reserve(m_se_hd_count * 4);
    for (auto& p : se_hd) {
        buf_hd.push_back(p[0]); buf_hd.push_back(p[1]);
        buf_hd.push_back(p[2]); buf_hd.push_back(0.f);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_se_hd);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 (GLsizeiptr)buf_hd.size() * sizeof(float), buf_hd.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void PointCloudBruteEroder::build_input_hash(uint32_t point_count) {
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

uint32_t PointCloudBruteEroder::erode(const std::vector<std::array<float, 3>>& input) {
    return dispatch_erode(input, m_prog_erode, /*write_all=*/false);
}

uint32_t PointCloudBruteEroder::erode_score(const std::vector<std::array<float, 3>>& input) {
    return dispatch_erode(input, m_prog_erode_score, /*write_all=*/true);
}

// Shared implementation used by both erode() and erode_score().
uint32_t PointCloudBruteEroder::dispatch_erode(
        const std::vector<std::array<float, 3>>& input,
        GLuint prog,
        bool write_all)
{
    if (input.empty() || m_se_ld_count == 0 || m_se_hd_count == 0) return 0;

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

    // ── 2. Build input hash from ALL points (must be complete before dispatch)
    build_input_hash(n);

    m_result_accum.clear();
    m_score_accum.clear();
    m_norm_accum.clear();

    glUseProgram(prog);
    glUniform1ui(glGetUniformLocation(prog, "u_table_size"),  m_cfg.table_size);
    glUniform1f (glGetUniformLocation(prog, "u_cell_size"),   m_cfg.cell_size);
    glUniform1f (glGetUniformLocation(prog, "u_min_dist_ld"), m_cfg.min_dist_ld * m_cfg.min_dist_ld);
    glUniform1f (glGetUniformLocation(prog, "u_min_dist_hd"), m_cfg.min_dist_hd * m_cfg.min_dist_hd);
    glUniform1ui(glGetUniformLocation(prog, "u_se_ld_count"), m_se_ld_count);
    glUniform1ui(glGetUniformLocation(prog, "u_se_hd_count"), m_se_hd_count);
    glUniform1ui(glGetUniformLocation(prog, "u_max_output"),  m_cfg.chunk_size);
    glUniform1f (glGetUniformLocation(prog, "u_angle_ld"),    m_cfg.angle_ld);
    glUniform1f (glGetUniformLocation(prog, "u_angle_hd"),    m_cfg.angle_hd);
    glUniform1i (glGetUniformLocation(prog, "u_z_axis"),    m_cfg.z_axis);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssbo_input_cells);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_ssbo_input_pts);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_ssbo_se_hd);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_ssbo_se_ld);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_ssbo_out_pts);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_ssbo_out_norm);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, m_ssbo_out_count);

    for (uint32_t offset = 0; offset < n; offset += m_cfg.chunk_size) {
        uint32_t this_chunk = std::min(m_cfg.chunk_size, n - offset);

        uint32_t zero = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_count);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &zero);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        glUniform1ui(glGetUniformLocation(prog, "u_num_input"),    this_chunk);
        glUniform1ui(glGetUniformLocation(prog, "u_chunk_offset"), offset);

        glDispatchCompute((this_chunk + 255) / 256, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // Hard erode: read back how many survivors were written this chunk.
        // Score erode: every point in the chunk is written, count == this_chunk.
        uint32_t chunk_out = this_chunk;
        if (!write_all) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_count);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &chunk_out);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
            chunk_out = std::min(chunk_out, m_cfg.chunk_size);
        }

        if (chunk_out > 0) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_pts);
            const float* pts_mapped = (const float*)glMapBufferRange(
                GL_SHADER_STORAGE_BUFFER, 0,
                (GLsizeiptr)chunk_out * 4 * sizeof(float), GL_MAP_READ_BIT);
            if (!pts_mapped) throw std::runtime_error("glMapBufferRange failed (pts)");
            if (write_all) {
                for (uint32_t i = 0; i < chunk_out; ++i)
                    m_score_accum.push_back({ pts_mapped[i*4], pts_mapped[i*4+1], pts_mapped[i*4+2], pts_mapped[i*4+3] });
            } else {
                for (uint32_t i = 0; i < chunk_out; ++i)
                    m_result_accum.push_back({ pts_mapped[i*4], pts_mapped[i*4+1], pts_mapped[i*4+2] });
            }
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_out_norm);
            const float* norm_mapped = (const float*)glMapBufferRange(
                GL_SHADER_STORAGE_BUFFER, 0,
                (GLsizeiptr)chunk_out * 4 * sizeof(float), GL_MAP_READ_BIT);
            if (!norm_mapped) throw std::runtime_error("glMapBufferRange failed (norm)");
            for (uint32_t i = 0; i < chunk_out; ++i)
                m_norm_accum.push_back({ norm_mapped[i*4], norm_mapped[i*4+1], norm_mapped[i*4+2] });
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }
    }

    m_result_count = write_all ? (uint32_t)m_score_accum.size()
                               : (uint32_t)m_result_accum.size();
    return m_result_count;
}

PointCloudBruteEroder::OrientationErosionResult PointCloudBruteEroder::get_result() const {
    return {m_result_accum, m_norm_accum};
}

PointCloudBruteEroder::OrientationErosionScoreResult PointCloudBruteEroder::get_result_with_scores() const {
    return {m_score_accum, m_norm_accum};
}

Point_set PointCloudBruteEroder::get_result_cgal() const {
    auto result = get_result();
    Point_set ps;
    auto norm_map = ps.add_normal_map().first;
    for (size_t i = 0; i < result.coord.size(); ++i) {
        auto it = ps.insert(Point(result.coord[i][0], result.coord[i][1], result.coord[i][2]));
        if (i < result.normal.size())
            ps.normal(*it) = Vector(result.normal[i][0], result.normal[i][1], result.normal[i][2]);
    }
    return ps;
}

Point_set PointCloudBruteEroder::get_result_cgal_with_scores() const {
    auto result = get_result_with_scores();
    Point_set ps;
    auto norm_map  = ps.add_normal_map().first;
    auto score_map = ps.add_property_map<float>("erosion_score", 0.f).first;
    for (size_t i = 0; i < result.coord.size(); ++i) {
        auto it = ps.insert(Point(result.coord[i][0], result.coord[i][1], result.coord[i][2]));
        score_map[*it] = result.coord[i][3];
        if (i < result.normal.size())
            ps.normal(*it) = Vector(result.normal[i][0], result.normal[i][1], result.normal[i][2]);
    }
    return ps;
}