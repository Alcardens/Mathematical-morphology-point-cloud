#include "gpu_spatial_hash.h"
#include "shaders.h"
#include <stdexcept>
#include <string>
#include <algorithm>

static void check_gl(const char* tag) {
    GLenum e = glGetError();
    if (e != GL_NO_ERROR)
        throw std::runtime_error(std::string(tag) + " GL error: " + std::to_string(e));
}

GLuint GPUSpatialHash::compile_compute(const char* src) {
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[4096]; glGetShaderInfoLog(sh, sizeof(buf), nullptr, buf);
        glDeleteShader(sh);
        throw std::runtime_error(std::string("Shader compile:\n") + buf);
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, sh); glDeleteShader(sh);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[4096]; glGetProgramInfoLog(prog, sizeof(buf), nullptr, buf);
        glDeleteProgram(prog);
        throw std::runtime_error(std::string("Shader link:\n") + buf);
    }
    return prog;
}

GPUSpatialHash::GPUSpatialHash(const SpatialHashConfig& cfg) : m_cfg(cfg) {
    if (cfg.table_size == 0 || (cfg.table_size & (cfg.table_size - 1)) != 0)
        throw std::invalid_argument("table_size must be a power of two");
    if (cfg.table_size < cfg.max_points * 2)
        std::cerr << "[GPUSpatialHash] WARNING: table_size < 2*max_points, "
                     "high load factor may degrade performance\n";

    // Each slot is uvec2 (8 bytes): key + point_index
    uint64_t cell_bytes  = (uint64_t)cfg.table_size * 4 * sizeof(uint32_t);
    uint64_t point_bytes = (uint64_t)cfg.max_points  * 4 * sizeof(float);

    std::cout << "[GPUSpatialHash] cells=" << cell_bytes  / (1024*1024) << " MB"
              << "  points=" << point_bytes / (1024*1024) << " MB\n";

    glGenBuffers(1, &m_ssbo_cells);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_cells);
    glBufferData(GL_SHADER_STORAGE_BUFFER, cell_bytes, nullptr, GL_DYNAMIC_COPY);
    check_gl("cells alloc");

    glGenBuffers(1, &m_ssbo_points);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_points);
    glBufferData(GL_SHADER_STORAGE_BUFFER, point_bytes, nullptr, GL_DYNAMIC_COPY);
    check_gl("points alloc");

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    m_prog_insert = compile_compute(SRC_INSERT);
    reset();
}

GPUSpatialHash::~GPUSpatialHash() {
    glDeleteBuffers(1, &m_ssbo_cells);
    glDeleteBuffers(1, &m_ssbo_points);
    glDeleteProgram(m_prog_insert);
}

void GPUSpatialHash::reset() {
    // Fill every key slot with EMPTY_KEY (0x7FFFFFFF)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_cells);
    uint32_t empty = 0x7FFFFFFFu;
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI,
                      GL_RED_INTEGER, GL_UNSIGNED_INT, &empty);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    m_point_count = 0;
}

void GPUSpatialHash::dispatch_insert(uint32_t start_offset, uint32_t count) {
    if (count == 0) return;
    glUseProgram(m_prog_insert);
    glUniform1ui(glGetUniformLocation(m_prog_insert, "u_table_size"),   m_cfg.table_size);
    glUniform1f (glGetUniformLocation(m_prog_insert, "u_cell_size"),    m_cfg.cell_size);
    glUniform1ui(glGetUniformLocation(m_prog_insert, "u_num_points"),   count);
    glUniform1ui(glGetUniformLocation(m_prog_insert, "u_start_offset"), start_offset);
    bind(0, 1);
    glDispatchCompute((count + 255) / 256, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

uint32_t GPUSpatialHash::append_candidates(GLuint cand_pts_ssbo,
                                            uint32_t candidate_count)
{
    if (candidate_count == 0 || is_full()) return 0;

    uint32_t n = std::min(candidate_count, remaining());

    // GPU copy: cand_pts_ssbo[0..n) -> m_ssbo_points[m_point_count..m_point_count+n)
    GLintptr   dst   = (GLintptr)m_point_count * 4 * sizeof(float);
    GLsizeiptr bytes = (GLsizeiptr)n * 4 * sizeof(float);
    glBindBuffer(GL_COPY_READ_BUFFER,  cand_pts_ssbo);
    glBindBuffer(GL_COPY_WRITE_BUFFER, m_ssbo_points);
    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, dst, bytes);
    glBindBuffer(GL_COPY_READ_BUFFER,  0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    check_gl("append_candidates copy");

    dispatch_insert(m_point_count, n);
    m_point_count += n;
    return n;
}

uint32_t GPUSpatialHash::upload_points(const std::vector<std::array<float, 3>>& pts) {
    uint32_t n = std::min((uint32_t)pts.size(), remaining());
    std::vector<float> buf;
    buf.reserve(n * 4);
    for (uint32_t i = 0; i < n; ++i) {
        buf.push_back(pts[i][0]); buf.push_back(pts[i][1]);
        buf.push_back(pts[i][2]); buf.push_back(0.f);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_points);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                    (GLintptr)(m_point_count * 4 * sizeof(float)),
                    (GLsizeiptr)buf.size() * sizeof(float), buf.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    dispatch_insert(m_point_count, n);
    m_point_count += n;
    return n;
}

void GPUSpatialHash::bind(GLuint cells_binding, GLuint points_binding) const {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, cells_binding,  m_ssbo_cells);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, points_binding, m_ssbo_points);
}

std::vector<std::array<float, 3>> GPUSpatialHash::download_points() const {
    if (m_point_count == 0) return {};
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo_points);
    const float* mapped = (const float*)glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0,
        (GLsizeiptr)m_point_count * 4 * sizeof(float), GL_MAP_READ_BIT);
    if (!mapped) throw std::runtime_error("glMapBufferRange failed");
    std::vector<std::array<float, 3>> out;
    out.reserve(m_point_count);
    for (uint32_t i = 0; i < m_point_count; ++i)
        out.push_back({ mapped[i*4], mapped[i*4+1], mapped[i*4+2] });
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return out;
}
