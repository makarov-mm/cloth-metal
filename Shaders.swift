// All Metal shaders for the cloth, compiled at runtime via
// device.makeLibrary(source:). Keeping the source here mirrors the C# project,
// where the GLSL also lived as string constants compiled at startup.

let metalSource = """
#include <metal_stdlib>
using namespace metal;

struct Params {
    float4 windDir;     // xyz used
    float dt;
    float time;
    float gravity;
    float damp;
    float baseBreeze;
    float curlStrength;
    float noiseFreq;
    float scrollSpeed;
    float windScale;
    float friction;     // tangential velocity kept on contact
    float floorY;       // ground plane height
    int   windOn;
    int   count;
};

struct SolveParams {
    int   offset;
    int   count;
    float dt;
    int   _pad;
};

struct Constraint {
    int   a;
    int   b;
    float rest;
    float compliance;
};

struct Lighting {
    float4 cam;
    float4 light;
    float4 front;
    float4 back;
};

// ---- curl-noise wind ------------------------------------------------------

inline float fade(float t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

inline float hash3(int3 q) {
    uint x = uint(q.x), y = uint(q.y), z = uint(q.z); // unsigned: defined wraparound
    uint h = x * 374761393u + y * 668265263u + z * 1442695040u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return float(h & 0xFFFFu) / 32767.5 - 1.0;
}

inline float vnoise(float3 P) {
    int3 i = int3(floor(P));
    float3 f = P - float3(i);
    float u = fade(f.x), v = fade(f.y), w = fade(f.z);
    float c000 = hash3(i + int3(0,0,0)), c100 = hash3(i + int3(1,0,0));
    float c010 = hash3(i + int3(0,1,0)), c110 = hash3(i + int3(1,1,0));
    float c001 = hash3(i + int3(0,0,1)), c101 = hash3(i + int3(1,0,1));
    float c011 = hash3(i + int3(0,1,1)), c111 = hash3(i + int3(1,1,1));
    float x00 = mix(c000, c100, u), x10 = mix(c010, c110, u);
    float x01 = mix(c001, c101, u), x11 = mix(c011, c111, u);
    return mix(mix(x00, x10, v), mix(x01, x11, v), w);
}

inline float3 pot(float3 s) {
    return float3(vnoise(s),
                  vnoise(s + float3(31.416, 17.073, 47.853)),
                  vnoise(s + float3(-19.34, 83.155, -5.926)));
}

inline float3 curl(float3 p) {
    float e = 0.12;
    float3 dx = float3(e,0,0), dy = float3(0,e,0), dz = float3(0,0,e);
    float3 pxp = pot(p+dx), pxm = pot(p-dx);
    float3 pyp = pot(p+dy), pym = pot(p-dy);
    float3 pzp = pot(p+dz), pzm = pot(p-dz);
    float inv = 1.0 / (2.0 * e);
    return float3(((pyp.z - pym.z) - (pzp.y - pzm.y)) * inv,
                  ((pzp.x - pzm.x) - (pxp.z - pxm.z)) * inv,
                  ((pxp.y - pxm.y) - (pyp.x - pym.x)) * inv);
}

inline float3 windField(float3 P, constant Params& Pp) {
    if (Pp.windOn == 0) return float3(0.0);
    float gust = 0.55 + 0.45 * sin(Pp.time * 0.7) * sin(Pp.time * 0.23 + 1.3);
    float3 s = P * Pp.noiseFreq + Pp.windDir.xyz * (Pp.time * Pp.scrollSpeed);
    return (Pp.windDir.xyz * Pp.baseBreeze + curl(s) * Pp.curlStrength) * (gust * Pp.windScale);
}

// ---- compute kernels ------------------------------------------------------

kernel void predict(device float4* pos        [[buffer(0)]],
                    device float4* prev        [[buffer(1)]],
                    constant Params& P         [[buffer(2)]],
                    uint gid                   [[thread_position_in_grid]]) {
    if (gid >= uint(P.count)) return;
    float4 cur4 = pos[gid];
    if (cur4.w == 0.0) return;               // pinned
    float3 cur = cur4.xyz;
    float3 vel = cur - prev[gid].xyz;
    float3 acc = float3(0.0, -P.gravity, 0.0) + windField(cur, P);
    float3 np = cur + vel * P.damp + acc * (P.dt * P.dt);
    pos[gid]  = float4(np, cur4.w);
    prev[gid] = float4(cur, 0.0);
}

kernel void clearLambda(device float* lam   [[buffer(0)]],
                        constant int& count [[buffer(1)]],
                        uint gid            [[thread_position_in_grid]]) {
    if (gid < uint(count)) lam[gid] = 0.0;
}

kernel void solve(device float4* pos              [[buffer(0)]],
                  device float* lam               [[buffer(1)]],
                  device const Constraint* cons   [[buffer(2)]],
                  constant SolveParams& S         [[buffer(3)]],
                  uint t                          [[thread_position_in_grid]]) {
    if (t >= uint(S.count)) return;
    int c = S.offset + int(t);
    Constraint con = cons[c];
    float4 PA = pos[con.a], PB = pos[con.b];
    float wa = PA.w, wb = PB.w, wsum = wa + wb;
    if (wsum == 0.0) return;
    float3 d = PA.xyz - PB.xyz;
    float len = length(d);
    if (len < 1e-6) return;
    float3 n = d / len;
    float C = len - con.rest;
    float at = con.compliance / (S.dt * S.dt);
    float dl = (-C - at * lam[c]) / (wsum + at);
    lam[c] += dl;
    float3 corr = dl * n;
    pos[con.a] = float4(PA.xyz + wa * corr, PA.w);
    pos[con.b] = float4(PB.xyz - wb * corr, PB.w);
}

kernel void collide(device float4* pos   [[buffer(0)]],
                    device float4* prev  [[buffer(1)]],
                    constant Params& P   [[buffer(2)]],
                    uint gid             [[thread_position_in_grid]]) {
    if (gid >= uint(P.count)) return;
    float4 cur = pos[gid];
    if (cur.w == 0.0) return;
    if (cur.y < P.floorY) {
        float3 p = float3(cur.x, P.floorY, cur.z);   // project onto the floor
        float3 vel = p - prev[gid].xyz;
        float3 vt = float3(vel.x, 0.0, vel.z);        // horizontal slide
        pos[gid] = float4(p, cur.w);
        prev[gid] = float4(prev[gid].xyz + vt * P.friction, 0.0); // friction
    }
}

kernel void buildMesh(device const float4* pos [[buffer(0)]],
                      device float* vtx        [[buffer(1)]],
                      constant int& gn         [[buffer(2)]],
                      uint gid                 [[thread_position_in_grid]]) {
    int M = gn * gn;
    if (gid >= uint(M)) return;
    int i = int(gid) % gn, j = int(gid) / gn;
    int il = max(i - 1, 0), ir = min(i + 1, gn - 1);
    int jl = max(j - 1, 0), jr = min(j + 1, gn - 1);
    float3 dxv = pos[j * gn + ir].xyz - pos[j * gn + il].xyz;
    float3 dyv = pos[jr * gn + i].xyz - pos[jl * gn + i].xyz;
    float3 nrm = cross(dyv, dxv);
    float l = length(nrm);
    nrm = (l > 1e-6) ? nrm / l : float3(0.0, 0.0, 1.0);
    float3 p = pos[gid].xyz;
    int o = int(gid) * 8;
    vtx[o]   = p.x; vtx[o+1] = p.y; vtx[o+2] = p.z;
    vtx[o+3] = nrm.x; vtx[o+4] = nrm.y; vtx[o+5] = nrm.z;
    vtx[o+6] = float(i) / float(gn - 1);
    vtx[o+7] = float(j) / float(gn - 1);
}

// ---- render ---------------------------------------------------------------

struct VSOut {
    float4 position [[position]];
    float3 wpos;
    float3 nrm;
    float2 uv;
};

vertex VSOut vmain(uint vid                 [[vertex_id]],
                   device const float* vtx  [[buffer(0)]],
                   constant float4x4& mvp   [[buffer(1)]]) {
    int o = int(vid) * 8;
    float3 p = float3(vtx[o], vtx[o+1], vtx[o+2]);
    float3 n = float3(vtx[o+3], vtx[o+4], vtx[o+5]);
    float2 uv = float2(vtx[o+6], vtx[o+7]);
    VSOut out;
    out.position = mvp * float4(p, 1.0);
    out.wpos = p;
    out.nrm = n;
    out.uv = uv;
    return out;
}

fragment float4 fmain(VSOut in                [[stage_in]],
                      constant Lighting& L    [[buffer(0)]],
                      texture2d<float> tex    [[texture(0)]],
                      sampler smp             [[sampler(0)]]) {
    float3 N = normalize(in.nrm);
    float3 V = normalize(L.cam.xyz - in.wpos);
    bool front = dot(N, V) >= 0.0;
    if (!front) N = -N;
    float3 Ld = normalize(L.light.xyz);
    float diff = max(dot(N, Ld), 0.0);
    float3 weave = tex.sample(smp, in.uv * 6.0).rgb;
    float3 base = (front ? L.front.xyz : L.back.xyz) * weave * 1.3;
    float3 col = base * (0.25 + 0.75 * diff);
    float3 H = normalize(Ld + V);
    col += float3(1.0) * pow(max(dot(N, H), 0.0), 40.0) * 0.25;
    col = col / (col + 0.85);
    return float4(pow(col, float3(0.85)), 1.0);
}

// ---- floor ----------------------------------------------------------------

vertex VSOut floorV(uint vid               [[vertex_id]],
                    device const float* v  [[buffer(0)]],
                    constant float4x4& mvp [[buffer(1)]]) {
    int o = int(vid) * 8;
    float3 p = float3(v[o], v[o+1], v[o+2]);
    VSOut out;
    out.position = mvp * float4(p, 1.0);
    out.wpos = p;
    out.nrm = float3(v[o+3], v[o+4], v[o+5]);
    out.uv = float2(v[o+6], v[o+7]);
    return out;
}

fragment float4 floorF(VSOut in             [[stage_in]],
                       constant Lighting& L [[buffer(0)]]) {
    float3 N = normalize(in.nrm);
    float3 Ld = normalize(L.light.xyz);
    float diff = max(dot(N, Ld), 0.0);
    float2 g = abs(fract(in.uv * 24.0) - 0.5);
    float line = smoothstep(0.0, 0.04, min(g.x, g.y));
    float3 base = mix(float3(0.13, 0.14, 0.17), float3(0.05, 0.05, 0.06), 1.0 - line);
    float3 col = base * (0.35 + 0.65 * diff);
    return float4(col, 1.0);
}
"""
