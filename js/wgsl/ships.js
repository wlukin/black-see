// Oil tankers: a procedural low-poly hull + superstructure that heaves, pitches and rolls
// with the low-passed wave field, plus the drone-strike game state (hits, fires, sinking)
// and the fire / smoke / explosion particles.
import { FRAME_WGSL, ATMO_CORE_WGSL, ENV_MIPS, N, CASCADES } from './common.js';
import { SKY_RENDER_FUNCS } from './sky.js';
import { RENDER_BINDINGS } from './ocean.js';

// Tanker hull in metres: x towards the bow, y up (0 = waterline), z to port/starboard.
export const SHIP_L = 245, SHIP_B = 42;
export const MAX_FIRES = 16;
export const HITS_TO_SINK = 4;
const DECK = 7, DRAFT = 14;
const SINK_TIME = 80;      // s from the fatal hit until the hull is gone
const SINK_DEPTH = 60;     // m, enough to take the funnel under

const halfB = (u) => SHIP_B / 2 * (u > 0.8 ? Math.sqrt(Math.max(0, 1 - ((u - 0.8) / 0.2) ** 2))
  : u < 0.1 ? 0.8 + 2 * u : 1);

// Flat-shaded triangle list, 10 floats per vertex: position, face normal, colour, window flag.
export function tankerMesh() {
  const out = [];
  const tri = (a, b, c, col, win = 0) => {
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    const l = Math.hypot(...n) || 1;
    for (const p of [a, b, c]) out.push(...p, n[0] / l, n[1] / l, n[2] / l, ...col, win);
  };
  const quad = (a, b, c, d, col, win) => { tri(a, b, c, col, win); tri(a, c, d, col, win); };
  const box = (x0, x1, y0, y1, z0, z1, col, win = 0) => {
    const p = (i) => [i & 1 ? x1 : x0, i & 2 ? y1 : y0, i & 4 ? z1 : z0];
    for (const f of [[0, 2, 6, 4], [1, 3, 7, 5], [2, 3, 7, 6], [0, 1, 3, 2], [4, 5, 7, 6]]) {
      quad(p(f[0]), p(f[1]), p(f[2]), p(f[3]), col, win);
    }
  };
  const BLACK = [0.035, 0.04, 0.045], RED = [0.30, 0.05, 0.035], DECKC = [0.20, 0.09, 0.06];
  const WHITE = [0.72, 0.71, 0.67], GREY = [0.32, 0.33, 0.33], FUNNEL = [0.05, 0.05, 0.06];

  // hull: lofted cross-sections; elliptical bow in plan, slightly narrowed stern, rising sheer
  const section = (u) => {
    const x = (u - 0.5) * SHIP_L, b = halfB(u), r = Math.min(3, b), top = DECK + 3 * Math.max(0, u - 0.85) / 0.15;
    const keel = -DRAFT * (u > 0.96 ? 1 - (u - 0.96) / 0.04 * 0.6 : 1);
    const side = [[b, top], [b, -0.5], [b, keel + r], [b - r * 0.3, keel + r * 0.3], [b - r, keel]];
    return [...side, ...side.slice().reverse().map(([z, y]) => [-z, y])].map(([z, y]) => [x, y, z]);
  };
  const ST = 40;
  let prev = section(0);
  const transom = prev;
  for (let i = 1; i <= ST; i++) {
    const cur = section(i / ST);
    for (let k = 0; k + 1 < cur.length; k++) {
      quad(prev[k], cur[k], cur[k + 1], prev[k + 1], (prev[k][1] + prev[k + 1][1]) / 2 > -0.5 ? BLACK : RED);
    }
    quad(prev[0], cur[0], cur[cur.length - 1], prev[prev.length - 1], DECKC);
    prev = cur;
  }
  for (let k = 1; k + 1 < transom.length; k++) tri(transom[0], transom[k], transom[k + 1], BLACK);

  // superstructure aft, funnel, pipe rack, manifold and a foremast
  const s = -SHIP_L / 2;
  box(s + 8, s + 30, DECK, DECK + 22, -16, 16, WHITE, 1);
  box(s + 24, s + 30, DECK + 22, DECK + 24, -SHIP_B / 2, SHIP_B / 2, WHITE);
  box(s + 3, s + 11, DECK, DECK + 27, -3.5, 3.5, FUNNEL);
  box(s + 3.2, s + 10.8, DECK + 23, DECK + 25, -3.6, 3.6, RED);
  box(s + 36, SHIP_L / 2 - 40, DECK, DECK + 1.6, -1.6, 1.6, GREY);
  box(-6, 6, DECK, DECK + 3, -14, 14, GREY);
  box(SHIP_L / 2 - 22, SHIP_L / 2 - 21, DECK + 3, DECK + 18, -0.5, 0.5, GREY);
  return new Float32Array(out);
}

// Is a point (tanker-local metres) inside the hull or the accommodation block?
function insideTanker(x, y, z) {
  if (Math.abs(x) > SHIP_L / 2 || y < -DRAFT) return false;
  if (y < DECK + 2 && Math.abs(z) < halfB(x / SHIP_L + 0.5)) return true;   // + deck gear and rails
  return x < -SHIP_L / 2 + 30 && Math.abs(z) < 16 && y < DECK + 27;
}

// Ships, hits, fires and sinking. Writes the GPU-side arrays `shipData` (8 floats per ship:
// x, z, heading, scale | sink depth, extra pitch, extra roll, -) and `fireData` (8 floats per
// fire: local x, y, z, ship | age, intensity, seed, -).
// routes: [start x, start z, heading deg (0 = +X), speed m/s, scale]
export function createFleet(routes) {
  const shipData = new Float32Array(routes.length * 8);
  const fireData = new Float32Array(MAX_FIRES * 8);
  const ships = [];
  const fires = [];
  const reset = () => {
    ships.length = 0;
    fires.length = 0;
    for (const [x, z, hdg, v, sc] of routes) {
      ships.push({ x, z, h: hdg * Math.PI / 180, v, sc, hits: 0, sinkT: -1, depth: 0, pitch: 0, roll: 0 });
    }
  };
  reset();

  function update(dt) {
    ships.forEach((s, i) => {
      if (s.hits) s.v *= Math.exp(-dt * (s.sinkT >= 0 ? 0.3 : 0.02));   // crippled ships lose way
      s.x += Math.cos(s.h) * s.v * dt;
      s.z += Math.sin(s.h) * s.v * dt;
      if (s.sinkT >= 0) s.sinkT += dt;
      const p = Math.max(0, Math.min(1, s.sinkT / SINK_TIME));
      s.depth = SINK_DEPTH * p * p;                        // slow at first, then goes down fast
      s.pitch = 0.22 * Math.min(1, p * 1.8);               // stern first: bow rises
      s.roll = 0.10 * Math.min(1, p * 2.5);
      shipData.set([s.x, s.z, s.h, s.sc, s.depth, s.pitch, s.roll, 0], i * 8);
    });
    fireData.fill(0);
    fires.forEach((f, i) => {
      f.age += dt;
      const s = ships[f.ship];
      // approximate height of the fire above the sea; it drowns as the hull goes under
      const wy = (f.y + f.x * s.pitch) * s.sc - s.depth;
      const I = Math.min(1, 0.3 + f.age / 6) * (0.6 + 0.15 * s.hits) * Math.max(0, Math.min(1, wy / 3 + 1));
      fireData.set([f.x, f.y, f.z, f.ship, f.age, I, f.seed, 0], i * 8);
    });
  }

  // Drone at world position (x, y, z): returns the index of the struck ship or -1.
  function strike(x, y, z) {
    for (let i = 0; i < ships.length; i++) {
      const s = ships[i];
      if (s.depth > 15) continue;
      const dx = x - s.x, dz = z - s.z, c = Math.cos(s.h), n = Math.sin(s.h);
      const lx = (dx * c + dz * n) / s.sc, lz = (-dx * n + dz * c) / s.sc, ly = (y + s.depth) / s.sc;
      if (!insideTanker(lx, ly, lz)) continue;
      s.hits++;
      if (s.hits >= HITS_TO_SINK && s.sinkT < 0) s.sinkT = 0;
      if (fires.length < MAX_FIRES) fires.push({ ship: i, x: lx, y: Math.max(ly, 0.5), z: lz, age: 0, seed: Math.random() * 1000 });
      return i;
    }
    return -1;
  }

  const state = (s) => s.sinkT >= SINK_TIME ? 'sunk' : s.sinkT >= 0 ? 'sinking' : s.hits ? 'burning' : 'intact';
  return { ships, shipData, fireData, fires, update, strike, reset, state };
}

// Shared by the hull and the particles: storage buffers + the wave-following ship pose.
const SHIP_POSE = /* wgsl */ `
struct Ship { pose: vec4f, sink: vec4f };   // x, z, heading, scale | depth, pitch, roll, -
struct Fire { p: vec4f, q: vec4f };         // local xyz, ship | age, intensity, seed, -
@group(0) @binding(4) var<storage, read> ships: array<Ship>;
@group(0) @binding(5) var<storage, read> fires: array<Fire, ${MAX_FIRES}>;
@group(0) @binding(6) var dispTex: texture_2d_array<f32>;
@group(0) @binding(7) var sampR: sampler;
@group(0) @binding(8) var<storage, read> sunVis: array<f32, 4>;
const SHIP_L: f32 = ${SHIP_L}.0;
const SHIP_B: f32 = ${SHIP_B}.0;
const MAX_FIRES: u32 = ${MAX_FIRES}u;

// wave height low-passed over ~25 m: a 245 m hull does not follow the short waves
fn waveH(xz: vec2f) -> f32 {
  var h = 0.0;
  for (var c = 0u; c < ${CASCADES}u; c++) {
    let lod = log2(max(25.0 * ${N}.0 / F.cascL[c >> 2u][c & 3u], 1.0));
    if (lod < 8.0) { h += textureSampleLevel(dispTex, sampR, xz * F.cascInvL[c >> 2u][c & 3u], c, lod).y; }
  }
  return h;
}

struct Pose { m: mat3x3f, o: vec3f, s: f32 };
fn shipPose(i: u32) -> Pose {
  let sp = ships[i];
  let fh = vec3f(cos(sp.pose.z), 0.0, sin(sp.pose.z));
  let sh = vec3f(-fh.z, 0.0, fh.x);
  let c = sp.pose.xy;
  let a = 0.35 * SHIP_L * sp.pose.w;
  let b = 0.4 * SHIP_B * sp.pose.w;
  let hb = waveH(c + fh.xz * a);
  let hs = waveH(c - fh.xz * a);
  let hp = waveH(c + sh.xz * b);
  let hq = waveH(c - sh.xz * b);
  // heavy hull: damp the response a little; sinking adds trim and list
  let f = normalize(fh + vec3f(0.0, 0.7 * (hb - hs) / (2.0 * a) + sp.sink.y, 0.0));
  var s = normalize(sh + vec3f(0.0, 0.7 * (hp - hq) / (2.0 * b) + sp.sink.z, 0.0));
  let u = normalize(cross(s, f));
  s = cross(f, u);
  var P: Pose;
  P.m = mat3x3f(f, u, s);
  P.o = vec3f(c.x, 0.25 * (hb + hs + hp + hq) - sp.sink.x, c.y);
  P.s = sp.pose.w;
  return P;
}
`;

export const SHIPS = FRAME_WGSL + ATMO_CORE_WGSL + RENDER_BINDINGS + SHIP_POSE + /* wgsl */ `
@group(0) @binding(9) var envCube: texture_cube<f32>;
` + SKY_RENDER_FUNCS + /* wgsl */ `
const ENV_MAXMIP: f32 = ${ENV_MIPS - 1}.0;

struct VO {
  @builtin(position) pos: vec4f,
  @location(0) rel: vec3f,
  @location(1) nrm: vec3f,
  @location(2) col: vec4f,
  @location(3) lp: vec3f,
  @location(4) @interpolate(flat) ship: u32,
};

@vertex fn vs(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) col: vec4f,
              @builtin(instance_index) ii: u32) -> VO {
  let P = shipPose(ii);
  let world = P.o + P.m * (p * P.s);
  var o: VO;
  o.rel = world - F.camPos.xyz;
  o.pos = F.viewProj * vec4f(o.rel, 1.0);
  o.nrm = P.m * n;
  o.col = col;
  o.lp = p;
  o.ship = ii;
  return o;
}

@fragment fn fs(in: VO) -> @location(0) vec4f {
  let dist = length(in.rel);
  let V = -in.rel / dist;
  var nrm = normalize(in.nrm);
  nrm *= sign(dot(nrm, V) + 1e-6);   // face the viewer regardless of triangle winding
  var alb = in.col.rgb;
  // rows of dark windows on the accommodation block
  if (in.col.w > 0.5 && fract(in.lp.y / 2.8) > 0.55 && fract(in.lp.x / 1.8 + in.lp.z / 1.8) > 0.35) {
    alb *= 0.15;
  }
  // strike damage: scorched paint around each fire, lit by its flames
  var burn = 0.0;
  var glow = 0.0;
  for (var j = 0u; j < MAX_FIRES; j++) {
    let fr = fires[j];
    if (fr.q.y <= 0.0 || u32(fr.p.w) != in.ship) { continue; }
    let d = distance(in.lp, fr.p.xyz);
    burn = max(burn, 1.0 - smoothstep(6.0, 28.0, d));
    glow += fr.q.y * exp(-d / 7.0) * (0.8 + 0.2 * sin(F.camPos.w * 23.0 + fr.q.z));
  }
  alb *= 1.0 - 0.85 * burn;
  let Ld = F.sunDir.xyz;
  let sunE = sunIlluminance() * sunVis[0];
  let amb = textureSampleLevel(envCube, sampC, nrm, ENV_MAXMIP - 1.0).rgb;
  var col = alb * (sunE * max(dot(nrm, Ld), 0.0) / PI + amb) + vec3f(3.0, 0.9, 0.2) * glow;
  // aerial perspective, as for the sea
  let T = exp(-max(F.fogExt.rgb, vec3f(F.fogExt.w)) * dist);
  let fogCol = skyAtmo(normalize(vec3f(-V.x, max(-V.y, 0.002), -V.z)));
  col = col * T + fogCol * (1.0 - T);
  return vec4f(col, 1.0);
}
`;

// Fire, smoke and impact fireball as camera-facing soft discs, one instance per fire.
// Particles are procedural in the fire's age, so nothing is simulated on the GPU.
export const FIRE_PARTICLES = 128;    // 32 fireball + 32 flame + 64 smoke
export const FIRES = FRAME_WGSL + ATMO_CORE_WGSL + RENDER_BINDINGS + SHIP_POSE + SKY_RENDER_FUNCS + /* wgsl */ `
struct PO {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) col: vec4f,      // premultiplied
};

fn rnd3(k: u32) -> vec3f {
  let h = vec3u(pcgHash(k), pcgHash(k ^ 0x9e3779b9u), pcgHash(k * 3u + 7u));
  return vec3f(h >> vec3u(8u)) / 16777216.0;
}

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) fi: u32) -> PO {
  let fr = fires[fi];
  let P = shipPose(u32(fr.p.w));
  let base = P.o + P.m * (fr.p.xyz * P.s);
  let k = vi / 6u;
  let corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
                                vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let cr = corners[vi % 6u];
  let r = rnd3(k + u32(fr.q.z) * 977u);
  let age = fr.q.x;
  let I = fr.q.y;
  let wind = vec3f(F.windP[0].x, 0.0, F.windP[0].y);
  var center = base;
  var size = 0.0;
  var col = vec4f(0.0);
  if (k < 32u) {
    // impact fireball: expands and fades within a couple of seconds
    let d = normalize(r * 2.0 - 1.0 + vec3f(0.0, 0.7, 0.0));
    center = base + d * 26.0 * (1.0 - exp(-age * 3.0)) + vec3f(0.0, 6.0 * age, 0.0);
    size = 7.0 + 9.0 * age;
    let a = exp(-age * 1.6) * step(age, 4.0);
    col = vec4f(vec3f(9.0, 3.2, 0.7) * a, a * 0.6);
  } else if (k < 64u) {
    // flames: short-lived licks rising from the hit
    let t = age - r.x * 1.2;
    let ph = fract(t / 1.2);
    center = base + vec3f((r.y - 0.5) * 10.0, ph * 16.0, (r.z - 0.5) * 10.0) * (0.5 + 0.6 * I);
    size = (2.5 + 5.0 * ph) * (0.5 + 0.7 * I);
    let a = sin(ph * PI) * I * step(0.0, t);
    col = vec4f(mix(vec3f(7.0, 3.0, 0.7), vec3f(2.0, 0.4, 0.06), ph) * a, a * 0.35);
  } else {
    // smoke: a dark plume climbing and bending downwind
    let T = 18.0;
    let t = age - r.x * T;
    let ph = fract(t / T);
    let spread = 4.0 + 34.0 * ph;
    center = base + vec3f(0.0, 10.0 + ph * 190.0, 0.0) + wind * (ph * ph * 160.0) +
             (r - 0.5) * vec3f(2.0, 1.0, 2.0) * spread;
    size = (8.0 + 60.0 * ph) * (0.6 + 0.4 * I);
    let a = 0.7 * smoothstep(0.0, 0.06, ph) * (1.0 - ph) * I * step(0.0, t);
    let lit = 0.05 * (skyAtmo(vec3f(0.0, 1.0, 0.0)) * 2.0 + sunIlluminance() * sunVis[0] * 0.08)
            + vec3f(1.5, 0.4, 0.08) * exp(-ph * 14.0) * I;
    col = vec4f(lit * a, a);
  }
  if (col.a <= 0.002) { size = 0.0; }
  let world = center + (cr.x * normalize(F.camRight.xyz) + cr.y * normalize(F.camUp.xyz)) * size;
  var o: PO;
  o.pos = F.viewProj * vec4f(world - F.camPos.xyz, 1.0);
  o.uv = cr;
  o.col = col;
  return o;
}

@fragment fn fs(in: PO) -> @location(0) vec4f {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  let m = (1.0 - r2) * (1.0 - r2);
  return in.col * m;
}
`;
