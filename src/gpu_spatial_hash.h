#pragma once
#include <GL/glew.h>
#include <vector>
#include <array>
#include <cstdint>
#include <iostream>

// Open-addressing spatial hash.
// Each table slot is a uvec2: { cell_key, point_index }.
// Empty slots have cell_key == 0xFFFFFFFF.
// Since cell_size == min_dist, at most one output point per cell is possible,
// so no slot ever needs to store more than one index — overflow is impossible
// as long as table_size > total output points (load factor < 1).
// Recommended: table_size = next_pow2(2 * max_points).
// Memory: table_size * 8 bytes.

struct SpatialHashConfig {
    float    cell_size;     // must equal min_dist
    float    min_dist;
    uint32_t table_size;    // power-of-two, >= 2 * max_points recommended
    uint32_t max_points;
};

class GPUSpatialHash {
public:
    explicit GPUSpatialHash(const SpatialHashConfig& cfg);
    ~GPUSpatialHash();

    // Copy accepted candidates into the output buffer and insert into hash.
    // Silently clamps to remaining capacity.
    // Returns number of points actually appended.
    uint32_t append_candidates(GLuint cand_pts_ssbo, uint32_t candidate_count);

    // Download all output points to CPU.
    std::vector<std::array<float, 3>> download_points() const;

    // Bind for external shaders.
    //   cells_binding  -> m_ssbo_cells   (Cells,     uvec2[table_size])
    //   points_binding -> m_ssbo_points  (OutputPts, vec4[max_points])
    void bind(GLuint cells_binding, GLuint points_binding) const;

    uint32_t point_count() const { return m_point_count; }
    uint32_t remaining()   const { return m_cfg.max_points - m_point_count; }
    bool     is_full()     const { return m_point_count >= m_cfg.max_points; }
    uint32_t upload_points(const std::vector<std::array<float, 3>>& pts);
    GLuint   points_ssbo() const { return m_ssbo_points; }
    const SpatialHashConfig& config() const { return m_cfg; }

    void reset();

private:
    void dispatch_insert(uint32_t start_offset, uint32_t count);

    SpatialHashConfig m_cfg;
    uint32_t          m_point_count = 0;

    GLuint m_ssbo_cells  = 0;   // uint[ table_size * 2 ]  (key + index per slot)
    GLuint m_ssbo_points = 0;   // float4[ max_points ]
    GLuint m_prog_insert = 0;

    static GLuint compile_compute(const char* src);
};

