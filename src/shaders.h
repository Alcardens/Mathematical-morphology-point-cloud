#pragma once

// INSERT
inline const char* SRC_INSERT = R"GLSL(
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) buffer Cells     { uint cells[];   };
layout(std430, binding = 1) buffer OutputPts { vec4 out_pts[]; };

uniform uint  u_table_size;
uniform float u_cell_size;
uniform uint  u_num_points;
uniform uint  u_start_offset;

const uint EMPTY_COORD = 0x7FFFFFFFu;

ivec3 cell_coord(vec3 p, float cs) { return ivec3(floor(p / cs)); }
uint  start_bucket(ivec3 c, uint table_size) {
    uint h = uint(c.x) * 2246822519u ^ uint(c.y) * 3266489917u ^ uint(c.z) * 668265263u;
    h ^= h >> 16u; h *= 0x45d9f3bu; h ^= h >> 16u;
    return h & (table_size - 1u);
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= u_num_points) return;

    uint  gidx = u_start_offset + id;
    ivec3 cc   = cell_coord(out_pts[gidx].xyz, u_cell_size);
    uint  slot = start_bucket(cc, u_table_size);

    for (uint probe = 0u; probe < u_table_size; ++probe) {
        uint idx = slot * 4u;
        // Try to claim this slot by swapping EMPTY_COORD into cx
        uint existing = atomicCompSwap(cells[idx], EMPTY_COORD, uint(cc.x));
        if (existing == EMPTY_COORD) {
            // We claimed the slot — write remaining fields
            cells[idx + 1u] = uint(cc.y);
            cells[idx + 2u] = uint(cc.z);
            cells[idx + 3u] = gidx;
            return;
        }
        // Slot already occupied — check if it's our own cell (duplicate insert)
        if (int(existing)    == cc.x &&
            int(cells[idx+1u]) == cc.y &&
            int(cells[idx+2u]) == cc.z) {
            cells[idx + 3u] = gidx;
            return;
        }
        slot = (slot + 1u) & (u_table_size - 1u);
    }
    // Table full
}
)GLSL";

// DILATE
inline const char* SRC_DILATE = R"GLSL(
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Cells     { uint cells[];    };
layout(std430, binding = 1) readonly buffer OutputPts { vec4 out_pts[];  };
layout(std430, binding = 2) readonly buffer InputPts  { vec4 in_pts[];   };
layout(std430, binding = 3) readonly buffer SE        { vec4 se_pts[];   };
layout(std430, binding = 4)          buffer CandPts   { vec4 cand_pts[]; };
layout(std430, binding = 5)          buffer CandCount { uint cand_count; };

uniform uint  u_table_size;
uniform float u_cell_size;
uniform float u_min_dist_sq;
uniform uint  u_num_input;
uniform uint  u_se_count;
uniform uint  u_max_cand;

const uint EMPTY_COORD = 0x7FFFFFFFu;

ivec3 cell_coord(vec3 p, float cs) { return ivec3(floor(p / cs)); }
uint  start_bucket(ivec3 c, uint table_size) {
    uint h = uint(c.x) * 2246822519u ^ uint(c.y) * 3266489917u ^ uint(c.z) * 668265263u;
    h ^= h >> 16u; h *= 0x45d9f3bu; h ^= h >> 16u;
    return h & (table_size - 1u);
}

bool lookup_cell(ivec3 nc, out vec4 found_pt) {
    uint slot = start_bucket(nc, u_table_size);
    for (uint probe = 0u; probe < u_table_size; ++probe) {
        uint idx = slot * 4u;
        uint cx  = cells[idx];
        if (cx == EMPTY_COORD) return false;          // empty slot — cell absent
        if (int(cx)           == nc.x &&
            int(cells[idx+1u]) == nc.y &&
            int(cells[idx+2u]) == nc.z) {
            found_pt = out_pts[cells[idx + 3u]];
            return true;
        }
        slot = (slot + 1u) & (u_table_size - 1u);
    }
    return false;
}

bool far_enough(vec3 c) {
    ivec3 base_cc = cell_coord(c, u_cell_size);
    for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dz = -1; dz <= 1; ++dz) {
        vec4 nb;
        if (lookup_cell(base_cc + ivec3(dx,dy,dz), nb)) {
            vec3 d = c - nb.xyz;
            if (dot(d,d) < u_min_dist_sq) return false;
        }
    }
    return true;
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= u_num_input * u_se_count) return;
    vec3 candidate = in_pts[id / u_se_count].xyz + se_pts[id % u_se_count].xyz;
    if (far_enough(candidate)) {
        uint slot = atomicAdd(cand_count, 1u);
        if (slot < u_max_cand)
            cand_pts[slot] = vec4(candidate, 0.0);
    }
}
)GLSL";

// DILATE_DENSITY
inline const char* SRC_DILATE_DENSITY = R"GLSL(
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Cells     { uint cells[];    };
layout(std430, binding = 1) readonly buffer OutputPts { vec4 out_pts[];  };
layout(std430, binding = 2) readonly buffer InputPts  { vec4 in_pts[];   };
layout(std430, binding = 3) readonly buffer SE        { vec4 se_pts[];   };
layout(std430, binding = 4)          buffer CandPts   { vec4 cand_pts[]; };
layout(std430, binding = 5)          buffer CandCount { uint cand_count; };

uniform uint  u_table_size;
uniform float u_cell_size;
uniform uint  u_num_input;
uniform uint  u_se_count;
uniform uint  u_max_cand;
uniform float u_min_dens;
uniform float u_max_dens;
uniform float u_increment;

const uint EMPTY_COORD = 0x7FFFFFFFu;

ivec3 cell_coord(vec3 p, float cs) { return ivec3(floor(p / cs)); }
uint  start_bucket(ivec3 c, uint table_size) {
    uint h = uint(c.x) * 2246822519u ^ uint(c.y) * 3266489917u ^ uint(c.z) * 668265263u;
    h ^= h >> 16u; h *= 0x45d9f3bu; h ^= h >> 16u;
    return h & (table_size - 1u);
}

bool lookup_cell(ivec3 nc, out vec4 found_pt) {
    uint slot = start_bucket(nc, u_table_size);
    for (uint probe = 0u; probe < u_table_size; ++probe) {
        uint idx = slot * 4u;
        uint cx  = cells[idx];
        if (cx == EMPTY_COORD) return false;
        if (int(cx)            == nc.x &&
            int(cells[idx+1u]) == nc.y &&
            int(cells[idx+2u]) == nc.z) {
            found_pt = out_pts[cells[idx + 3u]];
            return true;
        }
        slot = (slot + 1u) & (u_table_size - 1u);
    }
    return false;
}

bool far_enough(vec3 c, float density) {
    float min_dist_sq = density * density;
    ivec3 base_cc = cell_coord(c, u_cell_size);

    int shells = int(ceil(density / u_cell_size));

    for (int dx = -shells; dx <= shells; ++dx)
    for (int dy = -shells; dy <= shells; ++dy)
    for (int dz = -shells; dz <= shells; ++dz) {
        vec4 nb;
        if (lookup_cell(base_cc + ivec3(dx, dy, dz), nb)) {
            vec3 d = c - nb.xyz;
            if (dot(d, d) < min_dist_sq) return false;
        }
    }
    return true;
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= u_num_input * u_se_count) return;

    uint input_idx = id / u_se_count;
    uint se_idx    = id % u_se_count;

    vec4 input_pt = in_pts[input_idx];
    vec4 se_pt    = se_pts[se_idx];

    // Clip input density to [min_dens, max_dens]
    float input_density = clamp(input_pt.w, u_min_dens, u_max_dens);

    // Skip SE points whose density falls outside ± half increment of input density
    if (abs(se_pt.w - input_density) > u_increment * 0.5) return;

    vec3 candidate = input_pt.xyz + se_pt.xyz;

    if (far_enough(candidate, input_pt.w)) {
        uint slot = atomicAdd(cand_count, 1u);
        if (slot < u_max_cand)
            cand_pts[slot] = vec4(candidate, input_pt.w);
    }
}
)GLSL";

// ERODE
inline const char* SRC_ERODE = R"GLSL(
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Cells    { uint cells[];   };
layout(std430, binding = 1) readonly buffer InputPts { vec4 in_pts[];  };
layout(std430, binding = 2) readonly buffer SE       { vec4 se_pts[];  };
layout(std430, binding = 3)          buffer OutPts   { vec4 out_pts[]; };
layout(std430, binding = 4)          buffer OutCount { uint out_count; };

uniform uint  u_table_size;
uniform float u_cell_size;
uniform float u_min_dist_sq;
uniform uint  u_num_input;
uniform uint  u_se_count;
uniform uint  u_max_output;

const uint EMPTY_COORD = 0x7FFFFFFFu;

ivec3 cell_coord(vec3 p, float cs) { return ivec3(floor(p / cs)); }
uint  start_bucket(ivec3 c, uint table_size) {
    uint h = uint(c.x) * 2246822519u ^ uint(c.y) * 3266489917u ^ uint(c.z) * 668265263u;
    h ^= h >> 16u; h *= 0x45d9f3bu; h ^= h >> 16u;
    return h & (table_size - 1u);
}

bool has_neighbor(vec3 q) {
    ivec3 base_cc = cell_coord(q, u_cell_size);

    const int OFFSETS_X[27] = int[27](
         0,
         1, -1,  0,  0,  0,  0,
         1,  1, -1, -1,  1,  1, -1, -1,  0,  0,  0,  0,
         1,  1,  1,  1, -1, -1, -1, -1
    );
    const int OFFSETS_Y[27] = int[27](
         0,
         0,  0,  1, -1,  0,  0,
         1, -1,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,
         1,  1, -1, -1,  1,  1, -1, -1
    );
    const int OFFSETS_Z[27] = int[27](
         0,
         0,  0,  0,  0,  1, -1,
         0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,
         1, -1,  1, -1,  1, -1,  1, -1
    );

    for (int i = 0; i < 27; ++i) {
        ivec3 nc   = base_cc + ivec3(OFFSETS_X[i], OFFSETS_Y[i], OFFSETS_Z[i]);
        uint  slot = start_bucket(nc, u_table_size);
        for (uint probe = 0u; probe < u_table_size; ++probe) {
            uint idx = slot * 4u;
            uint cx  = cells[idx];
            if (cx == EMPTY_COORD) break;
            if (int(cx)            == nc.x &&
                int(cells[idx+1u]) == nc.y &&
                int(cells[idx+2u]) == nc.z) {
                vec3 d = q - in_pts[cells[idx + 3u]].xyz;
                if (dot(d, d) < u_min_dist_sq) return true;
                break;
            }
            slot = (slot + 1u) & (u_table_size - 1u);
        }
    }
    return false;
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= u_num_input) return;

    vec3 p = in_pts[id].xyz;

    for (uint j = 0u; j < u_se_count; ++j) {
        if (!has_neighbor(p + se_pts[j].xyz))
            return;
    }

    uint slot = atomicAdd(out_count, 1u);
    if (slot < u_max_output)
        out_pts[slot] = vec4(p, 0.0);
}
)GLSL";

// ERODE_SCORE
inline const char* SRC_ERODE_SCORE = R"GLSL(
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Cells    { uint cells[];   };
layout(std430, binding = 1) readonly buffer InputPts { vec4 in_pts[];  };
layout(std430, binding = 2) readonly buffer SE       { vec4 se_pts[];  };
layout(std430, binding = 3)          buffer OutPts   { vec4 out_pts[]; };
layout(std430, binding = 4)          buffer OutCount { uint out_count; };

uniform uint  u_table_size;
uniform float u_cell_size;
uniform float u_min_dist_sq;
uniform uint  u_num_input;
uniform uint  u_se_count;
uniform uint  u_max_output;

const uint EMPTY_COORD = 0x7FFFFFFFu;

ivec3 cell_coord(vec3 p, float cs) { return ivec3(floor(p / cs)); }
uint  start_bucket(ivec3 c, uint table_size) {
    uint h = uint(c.x) * 2246822519u ^ uint(c.y) * 3266489917u ^ uint(c.z) * 668265263u;
    h ^= h >> 16u; h *= 0x45d9f3bu; h ^= h >> 16u;
    return h & (table_size - 1u);
}

bool has_neighbor(vec3 q) {
    ivec3 base_cc = cell_coord(q, u_cell_size);

    const int OFFSETS_X[27] = int[27](
         0,
         1, -1,  0,  0,  0,  0,
         1,  1, -1, -1,  1,  1, -1, -1,  0,  0,  0,  0,
         1,  1,  1,  1, -1, -1, -1, -1
    );
    const int OFFSETS_Y[27] = int[27](
         0,
         0,  0,  1, -1,  0,  0,
         1, -1,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,
         1,  1, -1, -1,  1,  1, -1, -1
    );
    const int OFFSETS_Z[27] = int[27](
         0,
         0,  0,  0,  0,  1, -1,
         0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,
         1, -1,  1, -1,  1, -1,  1, -1
    );

    for (int i = 0; i < 27; ++i) {
        ivec3 nc   = base_cc + ivec3(OFFSETS_X[i], OFFSETS_Y[i], OFFSETS_Z[i]);
        uint  slot = start_bucket(nc, u_table_size);
        for (uint probe = 0u; probe < u_table_size; ++probe) {
            uint idx = slot * 4u;
            uint cx  = cells[idx];
            if (cx == EMPTY_COORD) break;
            if (int(cx)            == nc.x &&
                int(cells[idx+1u]) == nc.y &&
                int(cells[idx+2u]) == nc.z) {
                vec3 d = q - in_pts[cells[idx + 3u]].xyz;
                if (dot(d, d) < u_min_dist_sq) return true;
                break;
            }
            slot = (slot + 1u) & (u_table_size - 1u);
        }
    }
    return false;
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= u_num_input) return;

    vec3 p = in_pts[id].xyz;

    // Count how many SE offsets have a neighbour in the input hash.
    // No early exit — every offset must be tested to get an accurate ratio.
    uint matched = 0u;
    for (uint j = 0u; j < u_se_count; ++j) {
        if (has_neighbor(p + se_pts[j].xyz))
            matched++;
    }

    float score = float(matched) / float(u_se_count);

    // Every point is written; w holds the match ratio.
    uint slot = atomicAdd(out_count, 1u);
    if (slot < u_max_output)
        out_pts[slot] = vec4(p, score);
}
)GLSL";

// ADD
inline const char* SRC_ADD = R"GLSL(
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Cells       { uint cells[];    };
layout(std430, binding = 1) readonly buffer OriginalPts { vec4 or_pts[];   };
layout(std430, binding = 2) readonly buffer AdditionPts { vec4 ad_pts[];   };
layout(std430, binding = 3)          buffer OutPts      { vec4 out_pts[];  };
layout(std430, binding = 4)          buffer OutCount    { uint out_count;  };

uniform uint  u_table_size;
uniform float u_cell_size;
uniform float u_min_dist_sq;
uniform uint  u_num_addition;
uniform uint  u_max_output;

const uint EMPTY_COORD = 0x7FFFFFFFu;

ivec3 cell_coord(vec3 p, float cs) { return ivec3(floor(p / cs)); }
uint  start_bucket(ivec3 c, uint table_size) {
    uint h = uint(c.x) * 2246822519u ^ uint(c.y) * 3266489917u ^ uint(c.z) * 668265263u;
    h ^= h >> 16u; h *= 0x45d9f3bu; h ^= h >> 16u;
    return h & (table_size - 1u);
}

bool lookup_cell(ivec3 nc, out vec4 found_pt) {
    uint slot = start_bucket(nc, u_table_size);
    for (uint probe = 0u; probe < u_table_size; ++probe) {
        uint idx = slot * 4u;
        uint cx  = cells[idx];
        if (cx == EMPTY_COORD) return false;
        if (int(cx)            == nc.x &&
            int(cells[idx+1u]) == nc.y &&
            int(cells[idx+2u]) == nc.z) {
            found_pt = or_pts[cells[idx + 3u]];
            return true;
        }
        slot = (slot + 1u) & (u_table_size - 1u);
    }
    return false;
}

bool far_enough(vec3 c) {
    ivec3 base_cc = cell_coord(c, u_cell_size);
    for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dz = -1; dz <= 1; ++dz) {
        vec4 nb;
        if (lookup_cell(base_cc + ivec3(dx,dy,dz), nb)) {
            vec3 d = c - nb.xyz;
            if (dot(d,d) < u_min_dist_sq) return false;
        }
    }
    return true;
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= u_num_addition) return;

    vec3 p = ad_pts[id].xyz;

    if (far_enough(p)) {
        uint slot = atomicAdd(out_count, 1u);
        if (slot < u_max_output)
            out_pts[slot] = vec4(p, 0.0);
    }
}
)GLSL";

// DENSITY ESTIMATION
inline const char* SRC_DEN = R"GLSL(
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Cells   { uint  cells[];  };
layout(std430, binding = 1) readonly buffer InPts   { vec4  in_pts[]; };
layout(std430, binding = 2)          buffer OutDist { float dists[];  };

uniform uint  u_table_size;
uniform float u_cell_size;
uniform uint  u_num_points;
uniform int   u_max_shell;

const uint EMPTY_COORD = 0x7FFFFFFFu;
const int  K           = 5;

ivec3 cell_coord(vec3 p, float cs) { return ivec3(floor(p / cs)); }
uint  start_bucket(ivec3 c, uint table_size) {
    uint h = uint(c.x) * 2246822519u ^ uint(c.y) * 3266489917u ^ uint(c.z) * 668265263u;
    h ^= h >> 16u; h *= 0x45d9f3bu; h ^= h >> 16u;
    return h & (table_size - 1u);
}

bool lookup_cell(ivec3 nc, out vec3 found_pt) {
    uint slot = start_bucket(nc, u_table_size);
    for (uint probe = 0u; probe < u_table_size; ++probe) {
        uint idx = slot * 4u;
        uint cx  = cells[idx];
        if (cx == EMPTY_COORD) return false;
        if (int(cx)            == nc.x &&
            int(cells[idx+1u]) == nc.y &&
            int(cells[idx+2u]) == nc.z) {
            found_pt = in_pts[cells[idx + 3u]].xyz;
            return true;
        }
        slot = (slot + 1u) & (u_table_size - 1u);
    }
    return false;
}

void insert_best(inout float best[K], inout int found, float dist) {
    if (found < K || dist < best[K - 1]) {
        int pos = found < K ? found : K - 1;
        best[pos] = dist;
        found = min(found + 1, K);
        for (int i = pos; i > 0 && best[i] < best[i-1]; --i) {
            float tmp = best[i]; best[i] = best[i-1]; best[i-1] = tmp;
        }
    }
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= u_num_points) return;

    vec3  p    = in_pts[id].xyz;
    ivec3 base = cell_coord(p, u_cell_size);

    float best[K];
    for (int i = 0; i < K; ++i) best[i] = 1e38;
    int found = 0;

    for (int r = 0; r <= u_max_shell; ++r) {
        if (found == K && r > 1) {
            float min_shell_dist = float(r - 1) * u_cell_size;
            if (min_shell_dist * min_shell_dist > best[K - 1])
                break;
        }
        for (int dx = -r; dx <= r; ++dx)
        for (int dy = -r; dy <= r; ++dy)
        for (int dz = -r; dz <= r; ++dz) {
            if (max(max(abs(dx), abs(dy)), abs(dz)) != r) continue;
            vec3 nb;
            if (lookup_cell(base + ivec3(dx, dy, dz), nb)) {
                vec3  d    = p - nb;
                float dist = dot(d, d);
                if (dist > 0.0)
                    insert_best(best, found, dist);
            }
        }
    }

    float avg = 0.0;
    int   cnt = min(found, K);
    for (int i = 0; i < cnt; ++i)
        avg += sqrt(best[i]);
    if (cnt > 0) avg /= float(cnt);

    dists[id] = avg;
}
)GLSL";
