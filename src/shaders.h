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

// DILATE_ORIENTATION
inline const char* SRC_DILATE_ORIENTATION = R"GLSL(
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Cells     { uint cells[];    };
layout(std430, binding = 1) readonly buffer OutputPts { vec4 out_pts[];  };
layout(std430, binding = 2) readonly buffer InputPts  { vec4 in_pts[];   };
layout(std430, binding = 3) readonly buffer NormVecs  { vec4 norm_vec[]; };
layout(std430, binding = 4) readonly buffer SE        { vec4 se_pts[];   };
layout(std430, binding = 5)          buffer CandPts   { vec4 cand_pts[]; };
layout(std430, binding = 6)          buffer CandCount { uint cand_count; };

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

bool far_enough(vec3 c) {
    ivec3 base_cc = cell_coord(c, u_cell_size);
    for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dz = -1; dz <= 1; ++dz) {
        vec4 nb;
        if (lookup_cell(base_cc + ivec3(dx, dy, dz), nb)) {
            vec3 d = c - nb.xyz;
            if (dot(d, d) < u_min_dist_sq) return false;
        }
    }
    return true;
}

mat3 rotation_from_normal(vec3 n) {
    n = normalize(n);

    // Choose a reference vector not parallel to n to build the tangent
    vec3 ref = abs(n.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);

    vec3 tangent   = normalize(cross(ref, n));
    vec3 bitangent = cross(n, tangent);

    // Columns: tangent, normal, bitangent
    // SE local axes: x->tangent, y->normal, z->bitangent
    return mat3(tangent, n, bitangent);
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= u_num_input * u_se_count) return;

    uint input_idx = id / u_se_count;
    uint se_idx    = id % u_se_count;

    vec4 input_pt = in_pts[input_idx];
    vec3 normal   = norm_vec[input_idx].xyz;
    vec3 se_offset = se_pts[se_idx].xyz;

    // Rotate SE offset from local y-up space into world space aligned to normal
    mat3 rot      = rotation_from_normal(normal);
    vec3 candidate = input_pt.xyz + rot * se_offset;

    if (far_enough(candidate)) {
        uint slot = atomicAdd(cand_count, 1u);
        if (slot < u_max_cand)
            cand_pts[slot] = vec4(candidate, 0.0);
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
uniform uint  u_num_input;     // number of points in this chunk
uniform uint  u_se_count;
uniform uint  u_max_output;
uniform uint  u_chunk_offset;  // index of first point in this chunk within in_pts

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

    vec3 p = in_pts[u_chunk_offset + id].xyz;

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
uniform uint  u_num_input;     // number of points in this chunk
uniform uint  u_se_count;
uniform uint  u_max_output;
uniform uint  u_chunk_offset;  // index of first point in this chunk within in_pts

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

    vec3 p = in_pts[u_chunk_offset + id].xyz;

    uint matched = 0u;
    for (uint j = 0u; j < u_se_count; ++j) {
        if (has_neighbor(p + se_pts[j].xyz))
            matched++;
    }

    float score = float(matched) / float(u_se_count);

    uint slot = atomicAdd(out_count, 1u);
    if (slot < u_max_output)
        out_pts[slot] = vec4(p, score);
}
)GLSL";

// ERODE_DENSITY
inline const char* SRC_ERODE_DENSITY = R"GLSL(
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Cells    { uint cells[];   };
layout(std430, binding = 1) readonly buffer InputPts { vec4 in_pts[];  };
layout(std430, binding = 2) readonly buffer SE       { vec4 se_pts[];  };
layout(std430, binding = 3)          buffer OutPts   { vec4 out_pts[]; };
layout(std430, binding = 4)          buffer OutCount { uint out_count; };

uniform uint  u_table_size;
uniform float u_cell_size;
uniform uint  u_num_input;     // number of points in this chunk
uniform uint  u_se_count;
uniform uint  u_max_output;
uniform float u_min_dens;
uniform float u_max_dens;
uniform float u_increment;
uniform uint  u_chunk_offset;  // index of first point in this chunk within in_pts
uniform int   u_max_shell;

const uint EMPTY_COORD = 0x7FFFFFFFu;

ivec3 cell_coord(vec3 p, float cs) { return ivec3(floor(p / cs)); }
uint  start_bucket(ivec3 c, uint table_size) {
    uint h = uint(c.x) * 2246822519u ^ uint(c.y) * 3266489917u ^ uint(c.z) * 668265263u;
    h ^= h >> 16u; h *= 0x45d9f3bu; h ^= h >> 16u;
    return h & (table_size - 1u);
}

bool has_neighbor(vec3 q, float min_dist_sq, int shells) {
    shells = min(shells, u_max_shell);
    ivec3 base_cc = cell_coord(q, u_cell_size);

    for (int dx = -shells; dx <= shells; ++dx)
    for (int dy = -shells; dy <= shells; ++dy)
    for (int dz = -shells; dz <= shells; ++dz) {
        ivec3 nc   = base_cc + ivec3(dx, dy, dz);
        uint  slot = start_bucket(nc, u_table_size);
        for (uint probe = 0u; probe < u_table_size; ++probe) {
            uint idx = slot * 4u;
            uint cx  = cells[idx];
            if (cx == EMPTY_COORD) break;
            if (int(cx)            == nc.x &&
                int(cells[idx+1u]) == nc.y &&
                int(cells[idx+2u]) == nc.z) {
                vec3 d = q - in_pts[cells[idx + 3u]].xyz;
                if (dot(d, d) < min_dist_sq) return true;
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

    vec4 input_pt = in_pts[u_chunk_offset + id];
    vec3 p        = input_pt.xyz;

    // Clip input density to valid range
    float input_density = clamp(input_pt.w, u_min_dens, u_max_dens);
    float min_dist_sq    = input_density * input_density;
    int   shells         = int(ceil(input_density / u_cell_size));

    // Check each SE offset.
    for (uint j = 0u; j < u_se_count; ++j) {
        vec4 se_pt = se_pts[j];

        if (abs(se_pt.w - input_density) > u_increment * 0.5)
            continue;

        if (!has_neighbor(p + se_pt.xyz, min_dist_sq, shells))
            return;
    }

    uint slot = atomicAdd(out_count, 1u);
    if (slot < u_max_output)
        out_pts[slot] = vec4(p, 0);
}
)GLSL";

// ERODE_DENSITY_SCORE
inline const char* SRC_ERODE_DENSITY_SCORE = R"GLSL(
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Cells    { uint cells[];   };
layout(std430, binding = 1) readonly buffer InputPts { vec4 in_pts[];  };
layout(std430, binding = 2) readonly buffer SE       { vec4 se_pts[];  };
layout(std430, binding = 3)          buffer OutPts   { vec4 out_pts[]; };
layout(std430, binding = 4)          buffer OutCount { uint out_count; };

uniform uint  u_table_size;
uniform float u_cell_size;
uniform uint  u_num_input;     // number of points in this chunk
uniform uint  u_se_count;
uniform uint  u_max_output;
uniform float u_min_dens;
uniform float u_max_dens;
uniform float u_increment;
uniform uint  u_chunk_offset;  // index of first point in this chunk within in_pts
uniform int   u_max_shell;

const uint EMPTY_COORD = 0x7FFFFFFFu;

ivec3 cell_coord(vec3 p, float cs) { return ivec3(floor(p / cs)); }
uint  start_bucket(ivec3 c, uint table_size) {
    uint h = uint(c.x) * 2246822519u ^ uint(c.y) * 3266489917u ^ uint(c.z) * 668265263u;
    h ^= h >> 16u; h *= 0x45d9f3bu; h ^= h >> 16u;
    return h & (table_size - 1u);
}

bool has_neighbor(vec3 q, float min_dist_sq, int shells) {
    shells = min(shells, u_max_shell);
    ivec3 base_cc = cell_coord(q, u_cell_size);

    for (int dx = -shells; dx <= shells; ++dx)
    for (int dy = -shells; dy <= shells; ++dy)
    for (int dz = -shells; dz <= shells; ++dz) {
        ivec3 nc   = base_cc + ivec3(dx, dy, dz);
        uint  slot = start_bucket(nc, u_table_size);
        for (uint probe = 0u; probe < u_table_size; ++probe) {
            uint idx = slot * 4u;
            uint cx  = cells[idx];
            if (cx == EMPTY_COORD) break;
            if (int(cx)            == nc.x &&
                int(cells[idx+1u]) == nc.y &&
                int(cells[idx+2u]) == nc.z) {
                vec3 d = q - in_pts[cells[idx + 3u]].xyz;
                if (dot(d, d) < min_dist_sq) return true;
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

    vec4 input_pt = in_pts[u_chunk_offset + id];
    vec3 p        = input_pt.xyz;

    // Clip input density to valid range
    float input_density = clamp(input_pt.w, u_min_dens, u_max_dens);
    float min_dist_sq    = input_density * input_density;
    int   shells         = int(ceil(input_density / u_cell_size));

    // Check each SE offset.
    uint matched = 0u;
    uint applicable = 0u;
    for (uint j = 0u; j < u_se_count; ++j) {
        vec4 se_pt = se_pts[j];

        if (abs(se_pt.w - input_density) > u_increment * 0.5)
            continue;

        applicable++;
        if (has_neighbor(p + se_pt.xyz, min_dist_sq, shells))
            matched++;
    }

    float score = 0.0;

    if (applicable > 0u) {
        score = float(matched) / float(applicable);
    }

    uint slot = atomicAdd(out_count, 1u);
    if (slot < u_max_output)
        out_pts[slot] = vec4(p, score);
}
)GLSL";

// ERODE_ORIENTATION
inline const char* SRC_ERODE_ORIENTATION = R"GLSL(
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Cells    { uint cells[];    };
layout(std430, binding = 1) readonly buffer InputPts { vec4 in_pts[];   };
layout(std430, binding = 2) readonly buffer NormVecs { vec4 norm_vec[]; };
layout(std430, binding = 3) readonly buffer SE       { vec4 se_pts[];   };
layout(std430, binding = 4)          buffer OutPts   { vec4 out_pts[];  };
layout(std430, binding = 5)          buffer OutCount { uint out_count;  };

uniform uint  u_table_size;
uniform float u_cell_size;
uniform float u_min_dist_sq;
uniform uint  u_num_input;     // number of points in this chunk
uniform uint  u_se_count;
uniform uint  u_max_output;
uniform uint  u_chunk_offset;  // index of first point in this chunk within in_pts

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

mat3 rotation_from_normal(vec3 n) {
    n = normalize(n);

    // Choose a reference vector not parallel to n to build the tangent
    vec3 ref = abs(n.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);

    vec3 tangent   = normalize(cross(ref, n));
    vec3 bitangent = cross(n, tangent);

    // Columns: tangent, normal, bitangent
    // SE local axes: x->tangent, y->normal, z->bitangent
    return mat3(tangent, n, bitangent);
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= u_num_input) return;

    vec3 p = in_pts[u_chunk_offset + id].xyz;
    vec3 n = norm_vec[u_chunk_offset + id].xyz;

    mat3 rot = rotation_from_normal(n);

    for (uint j = 0u; j < u_se_count; ++j) {
        if (!has_neighbor(p + rot * se_pts[j].xyz))
            return;
    }

    uint slot = atomicAdd(out_count, 1u);
    if (slot < u_max_output)
        out_pts[slot] = vec4(p, 0.0);
}
)GLSL";

// ERODE_ORIENTATION_SCORE
inline const char* SRC_ERODE_ORIENTATION_SCORE = R"GLSL(
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Cells    { uint cells[];    };
layout(std430, binding = 1) readonly buffer InputPts { vec4 in_pts[];   };
layout(std430, binding = 2) readonly buffer NormVecs { vec4 norm_vec[]; };
layout(std430, binding = 3) readonly buffer SE       { vec4 se_pts[];   };
layout(std430, binding = 4)          buffer OutPts   { vec4 out_pts[];  };
layout(std430, binding = 5)          buffer OutCount { uint out_count;  };

uniform uint  u_table_size;
uniform float u_cell_size;
uniform float u_min_dist_sq;
uniform uint  u_num_input;     // number of points in this chunk
uniform uint  u_se_count;
uniform uint  u_max_output;
uniform uint  u_chunk_offset;  // index of first point in this chunk within in_pts

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

mat3 rotation_from_normal(vec3 n) {
    n = normalize(n);

    // Choose a reference vector not parallel to n to build the tangent
    vec3 ref = abs(n.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);

    vec3 tangent   = normalize(cross(ref, n));
    vec3 bitangent = cross(n, tangent);

    // Columns: tangent, normal, bitangent
    // SE local axes: x->tangent, y->normal, z->bitangent
    return mat3(tangent, n, bitangent);
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= u_num_input) return;

    vec3 p = in_pts[u_chunk_offset + id].xyz;
    vec3 n = norm_vec[u_chunk_offset + id].xyz;

    mat3 rot = rotation_from_normal(n);

    uint matched = 0u;
    for (uint j = 0u; j < u_se_count; ++j) {
        if (has_neighbor(p + rot * se_pts[j].xyz))
            matched++;
    }

    float score = float(matched) / float(u_se_count);

    uint slot = atomicAdd(out_count, 1u);
    if (slot < u_max_output)
        out_pts[slot] = vec4(p, score);
}
)GLSL";

// ERODE_ORIENTATION_BRUTE
inline const char* SRC_ERODE_ORIENTATION_BRUTE = R"GLSL(
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Cells    { uint cells[];     };
layout(std430, binding = 1) readonly buffer InputPts { vec4 in_pts[];    };
layout(std430, binding = 2) readonly buffer SE_HD    { vec4 se_hd_pts[]; };
layout(std430, binding = 3) readonly buffer SE_LD    { vec4 se_ld_pts[]; };
layout(std430, binding = 4)          buffer OutPts   { vec4 out_pts[];   };
layout(std430, binding = 5)          buffer OutNorm  { vec4 out_norm[];  };
layout(std430, binding = 6)          buffer OutCount { uint out_count;   };

uniform uint  u_table_size;
uniform float u_cell_size;
uniform float u_min_dist_ld;
uniform float u_min_dist_hd;
uniform uint  u_num_input;     // number of points in this chunk
uniform uint  u_se_ld_count;
uniform uint  u_se_hd_count;
uniform uint  u_max_output;
uniform uint  u_chunk_offset;  // index of first point in this chunk within in_pts
uniform float u_angle_ld;
uniform float u_angle_hd;
uniform bool  u_z_axis;

const uint EMPTY_COORD = 0x7FFFFFFFu;
const float PI          = 3.14159265359;
const float TWO_PI      = 6.28318530718;

ivec3 cell_coord(vec3 p, float cs) { return ivec3(floor(p / cs)); }
uint  start_bucket(ivec3 c, uint table_size) {
    uint h = uint(c.x) * 2246822519u ^ uint(c.y) * 3266489917u ^ uint(c.z) * 668265263u;
    h ^= h >> 16u; h *= 0x45d9f3bu; h ^= h >> 16u;
    return h & (table_size - 1u);
}

bool has_neighbor(vec3 q, float dist) {
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
                if (dot(d, d) < dist) return true;
                break;
            }
            slot = (slot + 1u) & (u_table_size - 1u);
        }
    }
    return false;
}

vec3 dir_from_angles(float theta, float phi) {
    float sp = sin(phi), cp = cos(phi);
    float st = sin(theta), ct = cos(theta);
    return vec3(sp * ct, cp, sp * st);
}

mat3 rotation_from_dir(vec3 n) {
    n = normalize(n);
    vec3 ref = abs(n.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent   = normalize(cross(ref, n));
    vec3 bitangent = cross(n, tangent);
    return mat3(tangent, n, bitangent);
}

bool match_ld(vec3 p, mat3 rot) {
    for (uint j = 0u; j < u_se_ld_count; ++j) {
        if (!has_neighbor(p + rot * se_ld_pts[j].xyz, u_min_dist_ld))
            return false;
    }
    return true;
}

bool match_hd(vec3 p, mat3 rot) {
    for (uint j = 0u; j < u_se_hd_count; ++j) {
        if (!has_neighbor(p + rot * se_hd_pts[j].xyz, u_min_dist_hd))
            return false;
    }
    return true;
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= u_num_input) return;

    vec3 p = in_pts[u_chunk_offset + id].xyz;

    int n_theta = int(ceil(TWO_PI / u_angle_ld));
    int n_phi   = u_z_axis ? 1 : int(ceil(PI / u_angle_ld));

    for (int it = 0; it < n_theta; ++it) {
        float theta = float(it) * u_angle_ld;
        for (int ip = 0; ip < n_phi; ++ip) {
            float phi = u_z_axis ? PI * 0.5 : float(ip) * u_angle_ld;
            vec3  dir = dir_from_angles(theta, phi);
            if(match_ld(p, rotation_from_dir(dir))) {
                int n_refine = int(ceil(u_angle_ld / u_angle_hd));

                for (int itr = -n_refine; itr <= n_refine; ++itr) {
                    float theta_r = theta + float(itr) * u_angle_hd;
                    theta_r = mod(theta_r + TWO_PI, TWO_PI);

                    int phi_start = u_z_axis ?  0 : -n_refine;
                    int phi_end   = u_z_axis ?  0 :  n_refine;

                    for (int ipr = phi_start; ipr <= phi_end; ++ipr) {
                        float phi_r = phi + float(ipr) * u_angle_hd;
                        // Clamp phi to [0, pi]
                        phi_r = clamp(phi_r, 0.0, PI);

                        vec3  dir_r = dir_from_angles(theta_r, phi_r);
                        if (match_hd(p, rotation_from_dir(dir_r))) {
                            uint slot = atomicAdd(out_count, 1u);
                            if (slot < u_max_output) {
                                out_pts [slot] = vec4(p, 0.0);
                                out_norm[slot] = vec4(dir_r, 0.0);
                            }
                            return;
                        }
                    }
                }
            }
        }
    }
}
)GLSL";

// ERODE_ORIENTATION_BRUTE_SCORE
inline const char* SRC_ERODE_ORIENTATION_BRUTE_SCORE = R"GLSL(
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Cells    { uint cells[];     };
layout(std430, binding = 1) readonly buffer InputPts { vec4 in_pts[];    };
layout(std430, binding = 2) readonly buffer SE_HD    { vec4 se_hd_pts[]; };
layout(std430, binding = 3) readonly buffer SE_LD    { vec4 se_ld_pts[]; };
layout(std430, binding = 4)          buffer OutPts   { vec4 out_pts[];   };
layout(std430, binding = 5)          buffer OutNorm  { vec4 out_norm[];  };
layout(std430, binding = 6)          buffer OutCount { uint out_count;   };

uniform uint  u_table_size;
uniform float u_cell_size;
uniform float u_min_dist_ld;
uniform float u_min_dist_hd;
uniform uint  u_num_input;     // number of points in this chunk
uniform uint  u_se_ld_count;
uniform uint  u_se_hd_count;
uniform uint  u_max_output;
uniform uint  u_chunk_offset;  // index of first point in this chunk within in_pts
uniform float u_angle_ld;
uniform float u_angle_hd;
uniform bool  u_z_axis;

const uint EMPTY_COORD = 0x7FFFFFFFu;
const float PI          = 3.14159265359;
const float TWO_PI      = 6.28318530718;

ivec3 cell_coord(vec3 p, float cs) { return ivec3(floor(p / cs)); }
uint  start_bucket(ivec3 c, uint table_size) {
    uint h = uint(c.x) * 2246822519u ^ uint(c.y) * 3266489917u ^ uint(c.z) * 668265263u;
    h ^= h >> 16u; h *= 0x45d9f3bu; h ^= h >> 16u;
    return h & (table_size - 1u);
}

bool has_neighbor(vec3 q, float dist) {
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
                if (dot(d, d) < dist) return true;
                break;
            }
            slot = (slot + 1u) & (u_table_size - 1u);
        }
    }
    return false;
}

vec3 dir_from_angles(float theta, float phi) {
    float sp = sin(phi), cp = cos(phi);
    float st = sin(theta), ct = cos(theta);
    return vec3(sp * ct, cp, sp * st);
}

mat3 rotation_from_dir(vec3 n) {
    n = normalize(n);
    vec3 ref = abs(n.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent   = normalize(cross(ref, n));
    vec3 bitangent = cross(n, tangent);
    return mat3(tangent, n, bitangent);
}

float score_ld(vec3 p, mat3 rot) {
    uint matched = 0u;
    for (uint j = 0u; j < u_se_ld_count; ++j) {
        if (has_neighbor(p + rot * se_ld_pts[j].xyz, u_min_dist_ld))
            matched++;
    }
    return float(matched) / float(u_se_ld_count);
}

float score_hd(vec3 p, mat3 rot) {
    uint matched = 0u;
    for (uint j = 0u; j < u_se_hd_count; ++j) {
        if (has_neighbor(p + rot * se_hd_pts[j].xyz, u_min_dist_hd))
            matched++;
    }
    return float(matched) / float(u_se_hd_count);
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= u_num_input) return;

    vec3 p = in_pts[u_chunk_offset + id].xyz;

    // ── Phase 1: coarse sweep with LD SE ─────────────────────────────────
    float best_ld_score = -1.0;
    float best_theta    =  0.0;
    float best_phi      = u_z_axis ? PI * 0.5 : 0.0;

    int n_theta = int(ceil(TWO_PI / u_angle_ld));
    int n_phi   = u_z_axis ? 1 : int(ceil(PI / u_angle_ld));

    for (int it = 0; it < n_theta; ++it) {
        float theta = float(it) * u_angle_ld;
        for (int ip = 0; ip < n_phi; ++ip) {
            float phi = u_z_axis ? PI * 0.5 : float(ip) * u_angle_ld;
            vec3  dir = dir_from_angles(theta, phi);
            float s   = score_ld(p, rotation_from_dir(dir));
            if (s > best_ld_score) {
                best_ld_score = s;
                best_theta    = theta;
                best_phi      = phi;
            }
        }
    }

    // ── Phase 2: refined sweep with HD SE around best LD direction ────────
    // Search a window of ± angle_ld around (best_theta, best_phi)
    // at angle_hd steps.

    float best_hd_score = -1.0;
    float ref_theta     = best_theta;
    float ref_phi       = best_phi;

    int n_refine = int(ceil(u_angle_ld / u_angle_hd));

    for (int it = -n_refine; it <= n_refine; ++it) {
        float theta = ref_theta + float(it) * u_angle_hd;
        // Wrap theta into [0, 2pi)
        theta = mod(theta + TWO_PI, TWO_PI);

        int phi_start = u_z_axis ?  0 : -n_refine;
        int phi_end   = u_z_axis ?  0 :  n_refine;

        for (int ip = phi_start; ip <= phi_end; ++ip) {
            float phi = ref_phi + float(ip) * u_angle_hd;
            // Clamp phi to [0, pi]
            phi = clamp(phi, 0.0, PI);

            vec3  dir = dir_from_angles(theta, phi);
            float s   = score_hd(p, rotation_from_dir(dir));
            if (s > best_hd_score) {
                best_hd_score = s;
                best_theta    = theta;
                best_phi      = phi;
            }
        }
    }

    // ── Write output ──────────────────────────────────────────────────────
    vec3 best_dir = dir_from_angles(best_theta, best_phi);

    uint slot = atomicAdd(out_count, 1u);
    if (slot < u_max_output) {
        out_pts [slot] = vec4(p, best_hd_score);
        out_norm[slot] = vec4(best_dir, 0.0);
    }
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

// INTERSECT
inline const char* SRC_INTERSECT = R"GLSL(
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Cells        { uint cells[];    };
layout(std430, binding = 1) readonly buffer IntersectPts { vec4 is_pts[];   };
layout(std430, binding = 2) readonly buffer OriginalPts  { vec4 or_pts[];   };
layout(std430, binding = 3)          buffer OutPts       { vec4 out_pts[];  };
layout(std430, binding = 4)          buffer OutCount     { uint out_count;  };

uniform uint  u_table_size;
uniform float u_cell_size;
uniform float u_min_dist_sq;
uniform uint  u_num_original;
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
                vec3 d = q - is_pts[cells[idx + 3u]].xyz;
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
    if (id >= u_num_original) return;

    vec3 p = or_pts[id].xyz;

    if (!has_neighbor(p)) return;

    uint slot = atomicAdd(out_count, 1u);
    if (slot < u_max_output)
        out_pts[slot] = vec4(p, 0.0);
}
)GLSL";

// SUBTRACT
inline const char* SRC_SUBTRACT = R"GLSL(
#version 430 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Cells        { uint cells[];    };
layout(std430, binding = 1) readonly buffer IntersectPts { vec4 is_pts[];   };
layout(std430, binding = 2) readonly buffer OriginalPts  { vec4 or_pts[];   };
layout(std430, binding = 3)          buffer OutPts       { vec4 out_pts[];  };
layout(std430, binding = 4)          buffer OutCount     { uint out_count;  };

uniform uint  u_table_size;
uniform float u_cell_size;
uniform float u_min_dist_sq;
uniform uint  u_num_original;
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
                vec3 d = q - is_pts[cells[idx + 3u]].xyz;
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
    if (id >= u_num_original) return;

    vec3 p = or_pts[id].xyz;

    if (has_neighbor(p)) return;

    uint slot = atomicAdd(out_count, 1u);
    if (slot < u_max_output)
        out_pts[slot] = vec4(p, 0.0);
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
