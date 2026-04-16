#include "point_cloud_erosion_orientation.h"
#include "shaders.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>

static void check_gl(const char* tag) {
    GLenum e = glGetError();
    if (e != GL_NO_ERROR)
        throw std::runtime_error(std::string(tag) + " GL error: " + std::to_string(e));
}

GLuint PointCloudOrientationEroder::compile_compute(const char* src) {
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

PointCloudOrientationEroder::PointCloudOrientationEroder(const OrientationErosionConfig& cfg) : m_cfg(cfg) {
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

    glGenBuffers(1, &m_ssbo_norm);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_norm);
    glBufferData(GL_SHADER_STORAGE_BUFFER, input_bytes, nullptr, GL_DYNAMIC_DRAW);
    check_gl("normals alloc");

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
    m_prog_erode  = compile_compute(SRC_ERODE_ORIENTATION);
    m_prog_erode_score = compile_compute(SRC_ERODE_ORIENTATION_SCORE);
}

PointCloudOrientationEroder::~PointCloudOrientationEroder() {
    glDeleteBuffers(1, &m_ssbo_input_cells);
    glDeleteBuffers(1, &m_ssbo_input_pts);
    glDeleteBuffers(1, &m_ssbo_norm);
    glDeleteBuffers(1, &m_ssbo_se);
    glDeleteBuffers(1, &m_ssbo_out_pts);
    glDeleteBuffers(1, &m_ssbo_out_count);
    glDeleteProgram(m_prog_insert);
    glDeleteProgram(m_prog_erode);
    glDeleteProgram(m_prog_erode_score);
}

void PointCloudOrientationEroder::set_structuring_element(
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

void PointCloudOrientationEroder::build_input_hash(uint32_t point_count) {
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

uint32_t PointCloudOrientationEroder::erode(const std::vector<std::array<float, 3>>& input,
                                            const std::vector<std::array<float, 3>>& normals) {
    return dispatch_erode(input, normals, m_prog_erode, /*write_all=*/false);
}

uint32_t PointCloudOrientationEroder::erode_score(const std::vector<std::array<float, 3>>& input,
                                                  const std::vector<std::array<float, 3>>& normals) {
    return dispatch_erode(input, normals, m_prog_erode_score, /*write_all=*/true);
}

// Shared implementation used by both erode() and erode_score().
uint32_t PointCloudOrientationEroder::dispatch_erode(
        const std::vector<std::array<float, 3>>& input,
        const std::vector<std::array<float, 3>>& normals,
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

    // ── 2. Build input hash from ALL points (must be complete before dispatch)
    build_input_hash(n);

    m_result_accum.clear();
    m_score_accum.clear();

    glUseProgram(prog);
    glUniform1ui(glGetUniformLocation(prog, "u_table_size"),  m_cfg.table_size);
    glUniform1f (glGetUniformLocation(prog, "u_cell_size"),   m_cfg.cell_size);
    glUniform1f (glGetUniformLocation(prog, "u_min_dist_sq"), m_cfg.min_dist * m_cfg.min_dist);
    glUniform1ui(glGetUniformLocation(prog, "u_se_count"),    m_se_count);
    glUniform1ui(glGetUniformLocation(prog, "u_max_output"),  m_cfg.chunk_size);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssbo_input_cells);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_ssbo_input_pts);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_ssbo_norm);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_ssbo_se);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_ssbo_out_pts);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, m_ssbo_out_count);

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
            const float* mapped = (const float*)glMapBufferRange(
                GL_SHADER_STORAGE_BUFFER, 0,
                (GLsizeiptr)chunk_out * 4 * sizeof(float), GL_MAP_READ_BIT);
            if (!mapped) throw std::runtime_error("glMapBufferRange failed (erode chunk)");
            if (write_all) {
                for (uint32_t i = 0; i < chunk_out; ++i)
                    m_score_accum.push_back({ mapped[i*4], mapped[i*4+1], mapped[i*4+2], mapped[i*4+3] });
            } else {
                for (uint32_t i = 0; i < chunk_out; ++i)
                    m_result_accum.push_back({ mapped[i*4], mapped[i*4+1], mapped[i*4+2] });
            }
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }
    }

    m_result_count = write_all ? (uint32_t)m_score_accum.size()
                               : (uint32_t)m_result_accum.size();
    return m_result_count;
}

std::vector<std::array<float, 3>> PointCloudOrientationEroder::get_result() const {
    return m_result_accum;
}

std::vector<std::array<float, 4>> PointCloudOrientationEroder::get_result_with_scores() const {
    return m_score_accum;
}

Point_set PointCloudOrientationEroder::get_result_cgal() const {
    auto pts = get_result();
    Point_set ps;
    for (auto& p : pts)
        ps.insert(Point(p[0], p[1], p[2]));
    return ps;
}

Point_set PointCloudOrientationEroder::get_result_cgal_with_scores() const {
    auto pts = get_result_with_scores();
    Point_set ps;
    auto score_map = ps.add_property_map<float>("erosion_score", 0.f).first;
    for (auto& p : pts) {
        auto it = ps.insert(Point(p[0], p[1], p[2]));
        score_map[*it] = p[3];
    }
    return ps;
}