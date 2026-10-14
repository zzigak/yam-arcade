// YAM arcade 3D viewer: replays the station exports converted by sync.py (data/index.json + data/<run>/manifest.json).
// Everything is in the station-local frame of export.py (z up, metres, quaternions w,x,y,z); the room places each
// station with layout.STATIONS[name].world = (x, y, yaw_deg). three.js runs Z-up here (camera.up = +z).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const DATA = 'data/';
const $ = (s) => document.querySelector(s);
const SRGB = THREE.SRGBColorSpace;

// ------------------------------------------------------------------ compositor constants (render/compose.py)
const KIND = {  // sRGB colour, roughness
  wood: [[0.72, 0.55, 0.38], 0.55], paint: [[0.80, 0.20, 0.18], 0.30], plastic: [[0.90, 0.90, 0.90], 0.35],
  rubber: [[0.12, 0.12, 0.12], 0.65], ceramic: [[0.93, 0.92, 0.89], 0.45], glass: [[0.965, 0.985, 0.978], 0.0],
  metal: [[0.80, 0.80, 0.82], 0.30], water: [[0.72, 0.90, 0.97], 0.0], fabric: [[0.55, 0.55, 0.60], 1.0],
  paper: [[0.95, 0.94, 0.90], 0.85], emissive: [[1.0, 0.85, 0.6], 0.5],
};
const FINISH = {
  mahjong: { top: 'wood', wood: [0.40, 0.26, 0.17], coat: 0.7, legs: 'black' },
  jars: { top: 'wood', wood: [0.80, 0.66, 0.48], legs: 'black' },
  cables: { top: 'white', wood: [0.80, 0.66, 0.48], legs: 'alu' },
  water: { top: 'wood', wood: [0.86, 0.75, 0.60], legs: 'alu' },
};
const DEFAULT_FINISH = { top: 'wood', wood: [0.78, 0.63, 0.45], legs: 'black' };
const OVERVIEW = { target: [0.30, 0.05, 0.45], az: 62.0, el: 35.0, dist: 7.4, lens: 35.0 };
const CLOSE = { eye: [0.66, 1.12, 1.60], target: [0.0, -0.10, 0.88], lens: 30.0 };
const OVER = { eye: [0.30, -1.55, 1.95], target: [0.0, 0.02, 0.78], lens: 32.0 };
const CLOSE_BY_STATION = { mahjong: { eye: [1.15, -0.80, 1.38], target: [-0.05, 0.08, 0.79], lens: 35.0 } };

// ------------------------------------------------------------------ state
const S = {
  index: null, layout: null, mode: 'room', runId: null, roomPick: {},  // station -> run id override
  t: 0, playing: true, speed: 1, loop: true, duration: 1,
  views: [],          // active RunView objects
  stations: {},       // name -> THREE.Group placed in the world (room) or at origin (run)
  dirty: true, loadToken: 0, camPreset: null,
};
window.__viewer = { S, ready: false };

// ------------------------------------------------------------------ renderer + scene
const canvas = $('#c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = SRGB;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.03, 200);
camera.up.set(0, 0, 1);
camera.position.set(6, -5, 4);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.screenSpacePanning = true;
controls.maxPolarAngle = Math.PI * 0.495;
controls.minDistance = 0.08;
controls.maxDistance = 40;
controls.addEventListener('change', () => { S.dirty = true; });
controls.addEventListener('start', () => { setCamButtons(null); });

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(renderer), 0.04).texture;
renderer.toneMapping = THREE.NoToneMapping;  // keeps the floor fog equal to the clear colour (no horizon seam)

const hemi = new THREE.HemisphereLight(0xffffff, 0xbdbdc2, 0.6);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xffffff, 1.55);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0005;
sun.shadow.normalBias = 0.02;
sun.shadow.radius = 4;
scene.add(sun, sun.target);
function aimSun(center, half) {
  sun.target.position.set(center.x, center.y, 0.6);
  sun.position.set(center.x + 2.0, center.y - 3.0, 7.0);
  const c = sun.shadow.camera;
  c.left = -half; c.right = half; c.top = half; c.bottom = -half; c.near = 0.5; c.far = 20;
  c.updateProjectionMatrix();
  sun.shadow.needsUpdate = true;
}

let floor;
function setupStudio(layout) {
  const bg = new THREE.Color().setRGB(...layout.BACKGROUND_RGB, SRGB);
  scene.background = bg;
  scene.fog = new THREE.Fog(bg, 14, 34);
  const fm = new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(...layout.FLOOR_RGB, SRGB), roughness: 0.85, metalness: 0, envMapIntensity: 0.3 });
  floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80), fm);
  floor.receiveShadow = true;
  floor.name = 'floor';
  scene.add(floor);
}

// ------------------------------------------------------------------ helpers
const srgb = (c) => new THREE.Color().setRGB(c[0], c[1], c[2], SRGB);
function timeAgo(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return `${Math.round(s)} s ago`;
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  if (s < 129600) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}
const fmtMB = (b) => `${(b / 1e6).toFixed(b < 1e7 ? 1 : 0)} MB`;

const progress = { total: 0, done: 0, label: '' };
function showLoading(on, label) {
  $('#loading').classList.toggle('hidden', !on);
  if (label) progress.label = label;
  updateLoading();
}
function updateLoading() {
  const f = progress.total > 0 ? Math.min(1, progress.done / progress.total) : 0;
  $('#loadFill').style.width = `${(f * 100).toFixed(1)}%`;
  $('#loadText').textContent = `${progress.label}${progress.total ? ` · ${fmtMB(progress.done)} of ~${fmtMB(progress.total)}` : ''}`;
}

async function fetchJSON(url) {
  const r = await fetch(url, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}
async function fetchBuf(url, expect = 0) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  if (!r.body || !r.body.getReader) { const b = await r.arrayBuffer(); progress.done += b.byteLength; updateLoading(); return b; }
  const len = +r.headers.get('content-length') || expect;
  const reader = r.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; progress.done += value.length; updateLoading();
  }
  const out = new Uint8Array(got); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  void len;
  return out.buffer;
}

// ------------------------------------------------------------------ GLB cache + materials
const gltfLoader = new GLTFLoader();
const glbCache = new Map();
function loadGLB(url, sizeHint = 0) {
  if (!glbCache.has(url)) {
    progress.total += sizeHint;
    glbCache.set(url, fetchBuf(DATA + url, sizeHint).then((buf) => new Promise((res, rej) => gltfLoader.parse(buf, '', (g) => res(g.scene), rej))));
  }
  return glbCache.get(url);
}

const matCache = new Map();
function makeMaterial(spec, src, opts = {}) {
  spec = spec || {};
  const kind = KIND[spec.kind] ? spec.kind : 'plastic';
  const key = JSON.stringify(spec) + '|' + (src ? src.uuid : '-') + '|' + JSON.stringify(opts);
  if (matCache.has(key)) return matCache.get(key);
  const [dcol, drough] = KIND[kind];
  const rough = spec.roughness ?? drough;
  const map = src && src.map ? src.map : null;
  if (map) map.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  let color;
  if (map) color = new THREE.Color(1, 1, 1);
  else if (spec.color) color = srgb(spec.color);
  else if (src && src.userData && src.userData.own && !['glass', 'water', 'metal'].includes(kind)) color = src.color.clone();
  else color = srgb(dcol);
  const vertexColors = !!(src && src.vertexColors && !spec.color);
  const common = { color, map, vertexColors, side: THREE.DoubleSide };
  let m;
  switch (kind) {
    case 'glass':
      m = new THREE.MeshPhysicalMaterial({ ...common, map: null, side: spec.thin ? THREE.DoubleSide : THREE.FrontSide,
        metalness: 0, roughness: Math.max(rough, 0.03), transmission: 1, thickness: 0.004, ior: spec.ior || 1.5,
        specularIntensity: 1, envMapIntensity: 1.2, transparent: true, opacity: 1 });
      break;
    case 'water': {
      const tint = srgb(spec.color || dcol);
      m = new THREE.MeshPhysicalMaterial({ color: new THREE.Color(1, 1, 1).lerp(tint, 0.55), metalness: 0,
        roughness: spec.roughness ?? 0.03, transmission: opts.droplet ? 0.7 : 0.92, thickness: opts.droplet ? 0.004 : 0.04,
        ior: 1.333, attenuationColor: tint, attenuationDistance: 0.12, side: THREE.DoubleSide, envMapIntensity: 1.2,
        transparent: true });
      break;
    }
    case 'metal':
      m = new THREE.MeshStandardMaterial({ ...common, metalness: 1, roughness: Math.max(rough, 0.18), envMapIntensity: 1.0 });
      break;
    case 'paint':
      m = new THREE.MeshPhysicalMaterial({ ...common, roughness: rough, clearcoat: 0.5, clearcoatRoughness: 0.08 });
      break;
    case 'ceramic':
      m = new THREE.MeshPhysicalMaterial({ ...common, roughness: rough, clearcoat: 1.0, clearcoatRoughness: 0.05 });
      break;
    case 'wood':
      m = new THREE.MeshPhysicalMaterial({ ...common, roughness: rough, clearcoat: spec.coat ?? 0.25, clearcoatRoughness: 0.18,
        map: map || woodTexture() });
      if (!map) m.color = srgb(spec.color || dcol);
      break;
    case 'fabric':
      m = new THREE.MeshPhysicalMaterial({ ...common, roughness: rough, sheen: 0.8, sheenRoughness: 0.45, sheenColor: color.clone() });
      break;
    case 'emissive':
      m = new THREE.MeshStandardMaterial({ ...common, roughness: rough, emissive: color.clone(), emissiveIntensity: Math.min(3, (spec.strength ?? 5) / 3) });
      break;
    default:  // plastic, rubber, paper
      m = new THREE.MeshStandardMaterial({ ...common, roughness: rough, metalness: 0 });
  }
  if (!['metal', 'glass', 'water'].includes(kind)) m.envMapIntensity = 0.45;
  m.userData.kind = kind;
  matCache.set(key, m);
  return m;
}

let _wood;
function woodTexture() {  // grey-scale plank grain, multiplied by the material colour
  if (_wood) return _wood;
  const c = document.createElement('canvas'); c.width = 512; c.height = 512;
  const g = c.getContext('2d');
  g.fillStyle = '#e8e8e8'; g.fillRect(0, 0, 512, 512);
  let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let y = 0; y < 512; y += 1) {
    const v = 0.94 + 0.03 * Math.sin(y * 0.37 + 4 * Math.sin(y * 0.021)) + 0.02 * (rnd() - 0.5);
    const k = Math.round(255 * Math.min(1, v));
    g.fillStyle = `rgb(${k},${k},${k})`; g.fillRect(0, y, 512, 1);
  }
  for (let i = 0; i < 900; i++) { g.fillStyle = `rgba(80,60,40,${0.03 * rnd()})`; g.fillRect(rnd() * 512, rnd() * 512, 20 + 60 * rnd(), 1); }
  _wood = new THREE.CanvasTexture(c);
  _wood.colorSpace = SRGB;
  _wood.wrapS = _wood.wrapT = THREE.RepeatWrapping;
  _wood.anisotropy = 4;
  return _wood;
}

// ------------------------------------------------------------------ table (render/compose.py build_table)
function buildTable(name, override) {
  const T = S.layout.TABLE;
  const fin = { ...(FINISH[name] || DEFAULT_FINISH), ...(override || {}) };
  const g = new THREE.Group(); g.name = `table_${name}`;
  const sx = T.size_x, sy = T.size_y, h = T.height, th = T.top_thickness, leg = T.leg_size, inset = T.leg_inset;
  let topMat;
  if (fin.top === 'white') topMat = new THREE.MeshStandardMaterial({ color: srgb([0.93, 0.93, 0.92]), roughness: 0.45 });
  else {
    const tex = woodTexture().clone(); tex.needsUpdate = true; tex.repeat.set(1, 0.6); tex.rotation = Math.PI / 2;
    topMat = new THREE.MeshPhysicalMaterial({ color: srgb(fin.wood), map: tex, roughness: 0.55, clearcoat: fin.coat ?? 0.35, clearcoatRoughness: 0.2, envMapIntensity: 0.45 });
  }
  const legMat = fin.legs === 'alu'
    ? new THREE.MeshStandardMaterial({ color: srgb([0.80, 0.81, 0.83]), metalness: 1, roughness: 0.3 })
    : new THREE.MeshStandardMaterial({ color: srgb([0.06, 0.06, 0.065]), roughness: 0.38 });
  const box = (lx, ly, lz, x, y, z, mat) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(lx, ly, lz), mat);
    m.position.set(x, y, z); m.castShadow = true; m.receiveShadow = true; m.userData.pick = `table (${name})`;
    g.add(m); return m;
  };
  box(sx, sy, th, 0, 0, h - th / 2, topMat);
  const xs = sx / 2 - inset - leg / 2, ys = sy / 2 - inset - leg / 2;
  for (const [a, b] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) box(leg, leg, h - th - 0.008, a * xs, b * ys, (h - th + 0.008) / 2, legMat);
  const ah = 0.06;
  for (const [cx, cy, lx, ly] of [[0, ys, 2 * xs, 0.02], [0, -ys, 2 * xs, 0.02], [xs, 0, 0.02, 2 * ys], [-xs, 0, 0.02, 2 * ys]])
    box(lx, ly, ah, cx, cy, h - th - ah / 2, legMat);
  return g;
}

function textSprite(lines) {
  const c = document.createElement('canvas'); const W = 512, H = 128; c.width = W; c.height = H;
  const g = c.getContext('2d');
  g.fillStyle = 'rgba(255,255,255,0.9)';
  const r = 24; g.beginPath(); g.roundRect ? g.roundRect(4, 4, W - 8, H - 8, r) : g.rect(4, 4, W - 8, H - 8); g.fill();
  g.fillStyle = '#080808'; g.textAlign = 'center';
  g.font = '500 46px Inter, Helvetica, Arial, sans-serif'; g.fillText(lines[0], W / 2, 58);
  g.fillStyle = 'rgba(8,8,8,0.6)'; g.font = '400 30px Inter, Helvetica, Arial, sans-serif'; g.fillText(lines[1] || '', W / 2, 100);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = SRGB;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthWrite: false, transparent: true, fog: false }));
  sp.scale.set(0.8, 0.2, 1);
  sp.renderOrder = 10;
  return sp;
}

// ------------------------------------------------------------------ rods: tube rebuilt in place every frame
const RADIAL = 8;
class RodTube {
  constructor(k, radius, mat, name) {
    this.k = k; this.r = radius;
    const nv = k * RADIAL;
    this.pos = new Float32Array(nv * 3); this.nrm = new Float32Array(nv * 3);
    const idx = [];
    for (let i = 0; i < k - 1; i++) for (let j = 0; j < RADIAL; j++) {
      const a = i * RADIAL + j, b = i * RADIAL + (j + 1) % RADIAL, c = a + RADIAL, d = b + RADIAL;
      idx.push(a, c, b, b, c, d);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('normal', new THREE.BufferAttribute(this.nrm, 3).setUsage(THREE.DynamicDrawUsage));
    geo.setIndex(idx);
    this.mesh = new THREE.Mesh(geo, mat);
    this.mesh.castShadow = true; this.mesh.receiveShadow = true; this.mesh.frustumCulled = false;
    this.mesh.userData.pick = name;
    const capGeo = new THREE.SphereGeometry(radius, 12, 8);
    this.caps = [new THREE.Mesh(capGeo, mat), new THREE.Mesh(capGeo, mat)];
    for (const c of this.caps) { c.castShadow = true; c.userData.pick = name; this.mesh.add(c); }
    this.cos = []; this.sin = [];
    for (let j = 0; j < RADIAL; j++) { this.cos.push(Math.cos(2 * Math.PI * j / RADIAL)); this.sin.push(Math.sin(2 * Math.PI * j / RADIAL)); }
    this.P = new Float32Array(k * 3);
  }
  update(P) {  // P: Float32Array k*3 (NaN-free)
    const k = this.k, r = this.r, pos = this.pos, nrm = this.nrm;
    let nx = 0, ny = 0, nz = 0, tx0 = 0, ty0 = 0, tz0 = 1;
    for (let i = 0; i < k; i++) {
      const a = Math.max(0, i - 1) * 3, b = Math.min(k - 1, i + 1) * 3;
      let tx = P[b] - P[a], ty = P[b + 1] - P[a + 1], tz = P[b + 2] - P[a + 2];
      let tl = Math.hypot(tx, ty, tz);
      if (tl < 1e-12) { tx = tx0; ty = ty0; tz = tz0; tl = 1; }
      tx /= tl; ty /= tl; tz /= tl;
      if (i === 0) {  // any normal perpendicular to the first tangent
        if (Math.abs(tz) < 0.9) { nx = -ty; ny = tx; nz = 0; } else { nx = 0; ny = -tz; nz = ty; }
      }
      const d = nx * tx + ny * ty + nz * tz;  // parallel transport: project the previous normal
      nx -= d * tx; ny -= d * ty; nz -= d * tz;
      let nl = Math.hypot(nx, ny, nz);
      if (nl < 1e-9) { if (Math.abs(tz) < 0.9) { nx = -ty; ny = tx; nz = 0; } else { nx = 0; ny = -tz; nz = ty; } nl = Math.hypot(nx, ny, nz); }
      nx /= nl; ny /= nl; nz /= nl;
      const bx = ty * nz - tz * ny, by = tz * nx - tx * nz, bz = tx * ny - ty * nx;
      const px = P[3 * i], py = P[3 * i + 1], pz = P[3 * i + 2];
      for (let j = 0; j < RADIAL; j++) {
        const c = this.cos[j], s = this.sin[j];
        const ox = c * nx + s * bx, oy = c * ny + s * by, oz = c * nz + s * bz;
        const o = 3 * (i * RADIAL + j);
        pos[o] = px + r * ox; pos[o + 1] = py + r * oy; pos[o + 2] = pz + r * oz;
        nrm[o] = ox; nrm[o + 1] = oy; nrm[o + 2] = oz;
      }
      tx0 = tx; ty0 = ty; tz0 = tz;
    }
    const g = this.mesh.geometry;
    g.attributes.position.needsUpdate = true; g.attributes.normal.needsUpdate = true;
    this.caps[0].position.set(P[0], P[1], P[2]);
    this.caps[1].position.set(P[3 * k - 3], P[3 * k - 2], P[3 * k - 1]);
  }
}

// ------------------------------------------------------------------ one run (export) in a station group
class RunView {
  constructor(id) { this.id = id; this.group = new THREE.Group(); this.items = []; this.frame = -1; this.ready = false; }

  async load() {
    const man = this.man = await fetchJSON(`${DATA}${this.id}/manifest.json`);
    this.group.name = `run_${this.id}`;
    const base = `${DATA}${this.id}/`;
    const fb = man.bytes || {};
    progress.total += (fb.frames || 0) - (fb.seq || 0) + (fb.meshes || 0);
    updateLoading();
    const jobs = [];
    if (man.files.traj) jobs.push(fetchBuf(base + man.files.traj).then((b) => { this.traj = new Float32Array(b); }));
    if (man.files.dyn) jobs.push(fetchBuf(base + man.files.dyn).then((b) => { this.dyn = new Uint16Array(b); }));
    const glbs = {};
    for (const o of man.objects) if (o.kind === 'rigid' && o.mesh && !glbs[o.mesh]) glbs[o.mesh] = loadGLB(o.mesh, 0);
    await Promise.all(jobs);
    const scenes = {};
    await Promise.all(Object.entries(glbs).map(async ([u, p]) => { try { scenes[u] = await p; } catch (e) { console.warn('mesh', u, e); } }));
    for (const o of man.objects) {
      try { this.addObject(o, scenes); } catch (e) { console.warn('object', o.name, e); }
    }
    this.ready = true;
  }

  addObject(o, scenes) {
    const kind = o.kind;
    if (kind === 'rigid') {
      const node = new THREE.Group(); node.name = o.name;
      const src = scenes[o.mesh];
      if (src) {
        src.updateMatrixWorld(true);
        src.traverse((ch) => {
          if (!ch.isMesh) return;
          const mat = makeMaterial(o.material, ch.material);
          const m = new THREE.Mesh(ch.geometry, mat);
          m.matrixAutoUpdate = false; m.matrix.copy(ch.matrixWorld);
          const tr = mat.userData.kind === 'glass' || mat.userData.kind === 'water';
          m.castShadow = !tr; m.receiveShadow = true;
          m.userData.pick = o.name;
          node.add(m);
        });
      }
      const sc = o.scale ?? 1;
      if (Array.isArray(sc)) node.scale.set(sc[0], sc[1], sc[2]); else node.scale.setScalar(sc);
      if (o.static_pose) {
        const [p, q] = o.static_pose;
        node.position.set(p[0], p[1], p[2]); node.quaternion.set(q[1], q[2], q[3], q[0]);
      }
      this.group.add(node);
      this.items.push({ o, node, kind });
    } else if (kind === 'rod') {
      const tube = new RodTube(o.k, o.radius, makeMaterial(o.material, null), o.name);
      this.group.add(tube.mesh);
      this.items.push({ o, tube, kind, buf: new Float32Array(o.k * 3) });
    } else if (kind === 'particles') {
      const droplet = (o.material || {}).kind === 'water';
      const geo = new THREE.IcosahedronGeometry(1, o.k > 8000 ? 0 : 1);
      const im = new THREE.InstancedMesh(geo, makeMaterial(o.material, null, { droplet }), o.k);
      im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      im.frustumCulled = false; im.castShadow = o.k <= 6000; im.receiveShadow = false;
      im.userData.pick = o.name;
      this.group.add(im);
      this.items.push({ o, im, kind });
    } else if (kind === 'meshseq') {
      const mat = makeMaterial(o.material, null);
      const mesh = new THREE.Mesh(new THREE.BufferGeometry(), mat);
      mesh.frustumCulled = false; mesh.userData.pick = o.name;
      mesh.castShadow = !['water', 'glass'].includes(mat.userData.kind); mesh.receiveShadow = true;
      this.group.add(mesh);
      this.items.push({ o, mesh, kind, cache: new Map(), shown: -1, frames: o.frames || [] });
    }
  }

  frameAt(t) {  // shared time (s) -> this run's output frame
    const man = this.man;
    const sf = Math.round(t * man.fps);
    if (man.frames_kept) {
      const fk = man.frames_kept; let lo = 0, hi = fk.length - 1;
      while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (fk[mid] <= sf) lo = mid; else hi = mid - 1; }
      return lo;
    }
    return Math.max(0, Math.min(man.n_frames - 1, sf));
  }
  srcFrame(f) { return this.man.frames_kept ? this.man.frames_kept[f] : f; }

  apply(f, force = false) {
    if (!this.ready || (f === this.frame && !force)) return;
    this.frame = f;
    const n = this.man.n_frames;
    for (const it of this.items) {
      const o = it.o;
      if (it.kind === 'rigid' && o.traj !== undefined && this.traj) {
        const b = o.traj + 7 * f, T = this.traj;
        it.node.position.set(T[b], T[b + 1], T[b + 2]);
        it.node.quaternion.set(T[b + 4], T[b + 5], T[b + 6], T[b + 3]);
      } else if (it.kind === 'rod' && this.dyn) {
        const k = o.k, D = this.dyn, b0 = o.dyn + f * k * 3, P = it.buf;
        let last = -1;
        for (let i = 0; i < k; i++) {
          const b = b0 + 3 * i;
          if (D[b] === 65535) { if (last >= 0) { P[3 * i] = P[3 * last]; P[3 * i + 1] = P[3 * last + 1]; P[3 * i + 2] = P[3 * last + 2]; } continue; }
          P[3 * i] = o.lo[0] + D[b] * o.step[0]; P[3 * i + 1] = o.lo[1] + D[b + 1] * o.step[1]; P[3 * i + 2] = o.lo[2] + D[b + 2] * o.step[2];
          last = i;
        }
        it.tube.update(P);
      } else if (it.kind === 'particles' && this.dyn) {
        const k = o.k, D = this.dyn, b0 = o.dyn + f * k * 3, A = it.im.instanceMatrix.array, r = o.radius;
        for (let i = 0; i < k; i++) {
          const b = b0 + 3 * i, m = 16 * i;
          const hide = D[b] === 65535;
          const s = hide ? 0 : r;
          A[m] = s; A[m + 1] = 0; A[m + 2] = 0; A[m + 3] = 0;
          A[m + 4] = 0; A[m + 5] = s; A[m + 6] = 0; A[m + 7] = 0;
          A[m + 8] = 0; A[m + 9] = 0; A[m + 10] = s; A[m + 11] = 0;
          A[m + 12] = hide ? 0 : o.lo[0] + D[b] * o.step[0];
          A[m + 13] = hide ? 0 : o.lo[1] + D[b + 1] * o.step[1];
          A[m + 14] = hide ? 0 : o.lo[2] + D[b + 2] * o.step[2];
          A[m + 15] = 1;
        }
        it.im.instanceMatrix.needsUpdate = true;
      } else if (it.kind === 'meshseq') {
        this.applySeq(it, f, n);
      }
    }
  }

  seqIndex(it, f) {  // latest available source frame <= this frame (as the compositor does)
    const fr = it.frames; if (!fr.length) return -1;
    const sf = this.srcFrame(f);
    let lo = 0, hi = fr.length - 1;
    if (fr[0] > sf) return fr[0];
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (fr[mid] <= sf) lo = mid; else hi = mid - 1; }
    return fr[lo];
  }
  loadSeq(it, j) {
    if (it.cache.has(j)) return it.cache.get(j);
    const url = `${DATA}${this.id}/${it.o.seq}${String(j).padStart(5, '0')}.bin`;
    const p = fetch(url).then((r) => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); }).then((buf) => {
      const h = new Uint32Array(buf, 0, 4); const nV = h[0], nF = h[1], ib = h[2];
      const q = new Float32Array(buf, 16, 6);
      const geo = new THREE.BufferGeometry();
      if (nV > 0) {
        const Q = new Uint16Array(buf, 40, nV * 3); const P = new Float32Array(nV * 3);
        for (let i = 0; i < nV; i++) for (let a = 0; a < 3; a++) P[3 * i + a] = q[a] + Q[3 * i + a] * q[3 + a];
        let off = 40 + nV * 6; off += (4 - (off % 4)) % 4;
        const I = ib === 2 ? new Uint16Array(buf, off, nF * 3) : new Uint32Array(buf, off, nF * 3);
        geo.setAttribute('position', new THREE.BufferAttribute(P, 3));
        geo.setIndex(new THREE.BufferAttribute(I.slice(), 1));
        geo.computeVertexNormals();
      }
      it.geoms = it.geoms || new Map(); it.geoms.set(j, geo);
      return geo;
    }).catch((e) => { console.warn('seq', url, e); it.cache.delete(j); return null; });
    it.cache.set(j, p);
    if (it.cache.size > 90) {  // evict the oldest
      const k0 = it.cache.keys().next().value; it.cache.delete(k0);
      const g = it.geoms && it.geoms.get(k0); if (g && it.mesh.geometry !== g) g.dispose(); if (it.geoms) it.geoms.delete(k0);
    }
    return p;
  }
  applySeq(it, f, n) {
    const j = this.seqIndex(it, f);
    if (j < 0) return;
    const show = (geo) => { if (geo && it.want === j) { it.mesh.geometry = geo; it.shown = j; S.dirty = true; } };
    it.want = j;
    const g = it.geoms && it.geoms.get(j);
    if (g) show(g); else this.loadSeq(it, j).then(show);
    for (let d = 1; d <= 8; d++) {  // prefetch ahead
      const fj = this.seqIndex(it, Math.min(n - 1, f + d * Math.max(1, Math.round(S.speed))));
      if (fj >= 0 && !it.cache.has(fj)) this.loadSeq(it, fj);
    }
  }

  pose(name) {
    const it = this.items.find((x) => x.o.name === name);
    if (!it || !it.node) return null;
    const wp = new THREE.Vector3(); it.node.getWorldPosition(wp);
    const q = it.node.quaternion;
    return { local: it.node.position.toArray(), quat_wxyz: [q.w, q.x, q.y, q.z], world: wp.toArray(), frame: this.frame, src_frame: this.srcFrame(this.frame) };
  }

  dispose() {
    this.group.traverse((o) => { if (o.isInstancedMesh) o.dispose(); if (o.geometry && o.userData.ownGeo) o.geometry.dispose(); });
    for (const it of this.items) {
      if (it.tube) it.tube.mesh.geometry.dispose();
      if (it.geoms) for (const g of it.geoms.values()) g.dispose();
    }
  }
}

// ------------------------------------------------------------------ world building
const world = new THREE.Group(); scene.add(world);
function clearWorld() {
  for (const v of S.views) v.dispose();
  S.views = [];
  world.clear();
  S.stations = {};
}
function stationMatrix(name) {
  const st = S.layout.STATIONS[name];
  const [x, y, yaw] = st ? st.world : [0, 0, 0];
  return { x, y, yaw: yaw * Math.PI / 180 };
}
function placeGroup(g, name, atOrigin) {
  if (atOrigin) return;
  const m = stationMatrix(name);
  g.position.set(m.x, m.y, 0); g.rotation.set(0, 0, m.yaw);
}

function runsOf(station) { return S.index.runs.filter((r) => r.station === station); }
function roomChoice(station) {
  if (S.roomPick[station]) { const r = S.index.runs.find((x) => x.id === S.roomPick[station]); if (r) return r; }
  const rs = runsOf(station);
  const done = rs.filter((r) => r.done).sort((a, b) => b.mtime - a.mtime);
  if (done.length) return done[0];
  return rs.slice().sort((a, b) => b.mtime - a.mtime)[0] || null;
}

async function buildRoom() {
  const tok = ++S.loadToken;
  clearWorld();
  progress.total = 0; progress.done = 0;
  showLoading(true, 'Loading the room');
  const names = Object.keys(S.layout.STATIONS);
  const chosen = {};
  for (const n of names) {
    const g = new THREE.Group(); g.name = `station_${n}`; placeGroup(g, n, false);
    const r = chosen[n] = roomChoice(n);
    g.add(buildTable(n, null));
    const st = S.layout.STATIONS[n];
    const lab = textSprite([st.title || n, r ? `${r.label === 'export' ? 'export' : r.label}${r.done ? ' · done' : ''}` : 'no export yet']);
    lab.position.set(0, 0.45, 1.7);
    g.add(lab);
    world.add(g); S.stations[n] = g;
  }
  aimSun(new THREE.Vector3(0, 0, 0), 4.2);
  S.duration = 1;
  setTitle(`<b>Room</b> · ${names.map((n) => `${S.layout.STATIONS[n].title}: ${chosen[n] ? chosen[n].label : '-'}`).join(' · ')}`);
  S.dirty = true;
  // load DONE / newest run of each station, mahjong first; each appears as soon as it is ready
  const order = names.filter((n) => chosen[n]).sort((a, b) => (chosen[b].done - chosen[a].done));
  await Promise.all(order.map(async (n) => {
    const v = new RunView(chosen[n].id);
    try { await v.load(); } catch (e) { console.warn('run', chosen[n].id, e); return; }
    if (tok !== S.loadToken) return;
    v.station = n;
    S.stations[n].add(v.group);
    S.views.push(v);
    S.duration = Math.max(S.duration, ...S.views.map((x) => x.man.duration));
    v.apply(v.frameAt(S.t), true);
    S.dirty = true;
  }));
  if (tok !== S.loadToken) return;
  showLoading(false);
  updateInfo();
  window.__viewer.ready = true;
}

async function buildRun(id) {
  const tok = ++S.loadToken;
  clearWorld();
  const meta = S.index.runs.find((r) => r.id === id);
  if (!meta) { showLoading(false); setTitle(`unknown run ${id}`); return; }
  progress.total = 0; progress.done = 0;
  showLoading(true, `Loading ${meta.station} / ${meta.label}`);
  const g = new THREE.Group(); g.name = `station_${meta.station}`;
  world.add(g); S.stations[meta.station] = g;
  const v = new RunView(id);
  setTitle(`<b>${S.layout.STATIONS[meta.station]?.title || meta.station}</b> · ${meta.label}${meta.done ? ' · done' : ''}`);
  try { await v.load(); } catch (e) { showLoading(true, `Failed: ${e.message}`); return; }
  if (tok !== S.loadToken) return;
  g.add(buildTable(meta.station, v.man.table));
  g.add(v.group);
  v.station = meta.station;
  S.views = [v];
  S.duration = Math.max(v.man.duration, 1 / v.man.fps);
  aimSun(new THREE.Vector3(0, -0.2, 0), 1.6);
  v.apply(v.frameAt(S.t), true);
  showLoading(false);
  updateInfo();
  S.dirty = true;
  window.__viewer.ready = true;
}

// ------------------------------------------------------------------ cameras
function lensFov(lens, eye, target) {  // compose.py: 36 mm sensor width -> keep the horizontal field on any aspect
  const aspect = camera.aspect;
  // portrait phones: keep only part of the preset's width so the table fills the screen instead of a thin strip
  const hfov = 2 * Math.atan(18 / lens) * (aspect < 1 ? Math.max(0.55, aspect) : 1);
  let vfov = 2 * Math.atan(Math.tan(hfov / 2) / aspect);
  const maxV = 72 * Math.PI / 180;
  let pull = 1;
  if (vfov > maxV) { const hNow = 2 * Math.atan(Math.tan(maxV / 2) * aspect); pull = Math.tan(hfov / 2) / Math.tan(hNow / 2); vfov = maxV; }
  return { fov: vfov * 180 / Math.PI, pull };
}
function setCamera(eye, target, lens) {
  const e = new THREE.Vector3(...eye), t = new THREE.Vector3(...target);
  const { fov, pull } = lensFov(lens, e, t);
  if (pull !== 1) e.sub(t).multiplyScalar(pull).add(t);
  camera.fov = fov; camera.updateProjectionMatrix();
  camera.position.copy(e); controls.target.copy(t); controls.update();
  S.dirty = true;
}
function toWorld(station, p, atOrigin) {
  if (atOrigin) return p.slice();
  const m = stationMatrix(station); const c = Math.cos(m.yaw), s = Math.sin(m.yaw);
  return [m.x + c * p[0] - s * p[1], m.y + s * p[0] + c * p[1], p[2]];
}
function presets() {
  const out = [];
  if (S.mode === 'room') {
    out.push({ id: 'overview', label: 'Overview', fn: () => {
      const o = OVERVIEW, az = o.az * Math.PI / 180, el = o.el * Math.PI / 180;
      const eye = [o.target[0] + o.dist * Math.cos(el) * Math.cos(az), o.target[1] + o.dist * Math.cos(el) * Math.sin(az), o.target[2] + o.dist * Math.sin(el)];
      setCamera(eye, o.target, o.lens);
    } });
    for (const n of Object.keys(S.layout.STATIONS)) {
      out.push({ id: n, label: S.layout.STATIONS[n].title || n, fn: () => {
        const v = S.views.find((x) => x.station === n);
        const p = { ...(CLOSE_BY_STATION[n] || CLOSE), ...((v && v.man.camera) || {}) };
        setCamera(toWorld(n, p.eye), toWorld(n, p.target), p.lens || 35);
      } });
    }
  } else {
    const v = S.views[0]; const n = v ? v.station : null;
    out.push({ id: 'close', label: 'Close-up', fn: () => {
      const p = { ...(CLOSE_BY_STATION[n] || CLOSE), ...((v && v.man.camera) || {}) };
      setCamera(p.eye, p.target, p.lens || 35);
    } });
    out.push({ id: 'over', label: 'Over shoulder', fn: () => setCamera(OVER.eye, OVER.target, OVER.lens) });
    out.push({ id: 'top', label: 'Top', fn: () => setCamera([0, -0.06, 2.35], [0, -0.05, 0.75], 30) });
    out.push({ id: 'side', label: 'Side', fn: () => setCamera([1.55, -0.1, 1.05], [0, -0.05, 0.85], 35) });
  }
  return out;
}
function renderCamButtons() {
  const nav = $('#cams'); nav.innerHTML = '';
  for (const p of presets()) {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = p.label; b.dataset.id = p.id;
    b.setAttribute('aria-pressed', 'false');
    b.onclick = () => { p.fn(); setCamButtons(p.id); };
    nav.appendChild(b);
  }
}
function setCamButtons(id) {
  S.camPreset = id;
  for (const b of document.querySelectorAll('#cams button')) b.setAttribute('aria-pressed', String(b.dataset.id === id));
}
function applyPreset(id) { const p = presets().find((x) => x.id === id) || presets()[0]; p.fn(); setCamButtons(p.id); }

// ------------------------------------------------------------------ UI
function setTitle(html) { $('#title').innerHTML = html; }
function updateInfo() {
  const el = $('#info');
  if (S.mode !== 'run' || !S.views[0]) { el.classList.add('hidden'); return; }
  const m = S.views[0].man, r = S.index.runs.find((x) => x.id === m.id) || {};
  const c = m.checks;
  const kinds = {}; for (const o of m.objects) kinds[o.kind] = (kinds[o.kind] || 0) + 1;
  el.innerHTML = `<b>${m.station} / ${m.label}</b>${m.done ? ' <span class="badge done">done</span>' : ''}
    <div class="row">${m.src_frames} frames at ${m.fps} fps${m.stride > 1 ? ` (shown every ${m.stride})` : ''} · ${m.duration.toFixed(1)} s</div>
    <div class="row">${Object.entries(kinds).map(([k, n]) => `${n} ${k}`).join(', ')}</div>
    ${c ? `<div class="row">checks ${c.n_pass}/${c.n}${c.fail.length ? ` · failing: ${c.fail.slice(0, 4).join(', ')}` : ''}</div>` : ''}
    <div class="row">exported ${timeAgo(m.mtime)} · ${m.source}</div>
    ${m.note ? `<div class="row">${escapeHtml(m.note.slice(0, 260))}${m.note.length > 260 ? '...' : ''}</div>` : ''}`;
  el.classList.remove('hidden');
}
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

function renderRunList() {
  const box = $('#runList'); box.innerHTML = '';
  const order = [...Object.keys(S.layout.STATIONS)];
  for (const r of S.index.runs) if (!order.includes(r.station)) order.push(r.station);
  for (const st of order) {
    const rs = runsOf(st).sort((a, b) => b.mtime - a.mtime);
    const head = document.createElement('div'); head.className = 'st';
    const title = S.layout.STATIONS[st]?.title || st;
    head.innerHTML = `<h2>${escapeHtml(title)}</h2><span class="muted">${rs.length ? `${rs.length} run${rs.length > 1 ? 's' : ''}` : 'no export yet'}${S.layout.STATIONS[st] ? '' : ' · not in the room'}</span>`;
    box.appendChild(head);
    const inRoom = S.layout.STATIONS[st] ? roomChoice(st) : null;
    for (const r of rs) {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'run';
      b.setAttribute('aria-current', String(S.mode === 'run' && S.runId === r.id));
      const c = r.checks;
      const chk = c ? `<span class="badge${c.all_ok ? '' : ' fail'}">checks ${c.n_pass}/${c.n}</span>` : '';
      const size = r.bytes ? (r.bytes.frames + r.bytes.meshes) : 0;
      b.innerHTML = `<div class="l1"><span class="name">${escapeHtml(r.label)}</span>${r.done ? '<span class="badge done">done</span>' : ''}${chk}
        ${S.layout.STATIONS[st] ? `<span class="pill small room" role="button" tabindex="0" aria-pressed="${inRoom && inRoom.id === r.id}" title="Show this run at its table in the room">${inRoom && inRoom.id === r.id ? 'in room' : 'use in room'}</span>` : ''}</div>
        <div class="l2">${timeAgo(r.mtime)} · ${r.src_frames} frames, ${r.duration.toFixed(1)} s · ${r.kinds.join(', ')}${r.robot ? ', robot' : ''} · ${fmtMB(size)}</div>`;
      b.onclick = (ev) => {
        if (ev.target.classList.contains('room')) {
          S.roomPick[st] = r.id; closeRuns(); setMode('room', null, true); return;
        }
        closeRuns(); setMode('run', r.id, true);
      };
      box.appendChild(b);
    }
  }
  $('#syncAge').textContent = `synced ${timeAgo(S.index.generated)}`;
}
function openRuns() { renderRunList(); $('#runs').hidden = false; $('#runsBtn').setAttribute('aria-expanded', 'true'); }
function closeRuns() { $('#runs').hidden = true; $('#runsBtn').setAttribute('aria-expanded', 'false'); }
$('#runsBtn').onclick = () => ($('#runs').hidden ? openRuns() : closeRuns());
$('#runsClose').onclick = closeRuns;
for (const b of document.querySelectorAll('.seg button')) b.onclick = () => {
  if (b.dataset.mode === 'run') {
    const id = S.runId || (roomChoice('mahjong') || S.index.runs[0] || {}).id;
    if (id) setMode('run', id, true); else openRuns();
  } else setMode('room', null, true);
};

function writeHash() {
  const p = new URLSearchParams();
  if (S.mode === 'run') p.set('run', S.runId); else p.set('room', '1');
  for (const [k, v] of Object.entries(S.roomPick)) p.set(`pick_${k}`, v);
  history.replaceState(null, '', `#${p.toString()}`);
}
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  S.roomPick = {};
  for (const [k, v] of p) if (k.startsWith('pick_')) S.roomPick[k.slice(5)] = v;
  const t = parseFloat(p.get('t')); if (Number.isFinite(t)) S.t = t;
  if (p.get('play') === '0') setPlaying(false);
  if (p.get('run')) return { mode: 'run', id: p.get('run'), cam: p.get('cam') };
  return { mode: 'room', id: null, cam: p.get('cam') };
}

async function setMode(mode, id, userAction, cam) {
  S.mode = mode; S.runId = mode === 'run' ? id : null;
  for (const b of document.querySelectorAll('.seg button')) b.setAttribute('aria-pressed', String(b.dataset.mode === mode));
  if (userAction) S.t = 0;
  writeHash();
  window.__viewer.ready = false;
  renderCamButtons();
  applyPreset(cam || (mode === 'room' ? 'overview' : 'close'));
  if (mode === 'room') await buildRoom(); else await buildRun(id);
  renderCamButtons();
  applyPreset(cam || S.camPreset || (mode === 'room' ? 'overview' : 'close'));
  $('#reloadBtn').classList.add('hidden');
}

// timeline
const scrub = $('#scrub');
function setPlaying(p) {
  S.playing = p;
  $('#playBtn').innerHTML = p ? '&#10074;&#10074;' : '&#9654;';
  $('#playBtn').setAttribute('aria-label', p ? 'Pause' : 'Play');
}
$('#playBtn').onclick = () => { if (!S.playing && S.t >= S.duration - 1e-6) S.t = 0; setPlaying(!S.playing); };
$('#speed').onchange = (e) => { S.speed = parseFloat(e.target.value); };
$('#loopBtn').onclick = () => { S.loop = !S.loop; $('#loopBtn').setAttribute('aria-pressed', String(S.loop)); };
let scrubbing = false;
scrub.addEventListener('input', () => { scrubbing = true; S.t = (scrub.value / 1000) * S.duration; applyTime(); });
scrub.addEventListener('change', () => { scrubbing = false; });
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
  if (e.code === 'Space') { e.preventDefault(); $('#playBtn').click(); }
  const fps = S.views[0] ? S.views[0].man.fps : 30;
  if (e.code === 'ArrowRight') { S.t = Math.min(S.duration, S.t + 1 / fps); setPlaying(false); applyTime(); }
  if (e.code === 'ArrowLeft') { S.t = Math.max(0, S.t - 1 / fps); setPlaying(false); applyTime(); }
});

function applyTime() {
  for (const v of S.views) v.apply(v.frameAt(S.t));
  if (!scrubbing) scrub.value = String(Math.round(1000 * (S.duration > 0 ? S.t / S.duration : 0)));
  const v0 = S.mode === 'run' ? S.views[0] : null;
  let fr = '';
  if (v0) fr = ` · f ${v0.srcFrame(v0.frame)}/${v0.man.src_frames - 1}`;
  $('#clock').textContent = `${S.t.toFixed(2)} / ${S.duration.toFixed(1)} s${fr}`;
  // events (mahjong logs pick/transport/release with times)
  const ev = $('#event');
  let msg = '';
  if (v0 && v0.man.events && v0.man.events.length) {
    for (const e of v0.man.events) if (e.t <= S.t + 1e-6) msg = e.msg;
  }
  ev.classList.toggle('hidden', !msg);
  if (msg && ev.textContent !== msg) ev.textContent = msg;
  S.dirty = true;
}

// picking
const ray = new THREE.Raycaster();
let down = null;
canvas.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; });
canvas.addEventListener('pointerup', (e) => {
  if (!down) return;
  const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
  const quick = performance.now() - down.t < 450;
  down = null;
  if (moved > 6 || !quick) return;
  const rect = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const hits = ray.intersectObject(world, true);
  const el = $('#pick');
  const h = hits.find((x) => x.object.userData.pick && x.object.visible);
  if (!h) { el.classList.add('hidden'); return; }
  let st = ''; let p = h.object; while (p && !p.name.startsWith('station_')) p = p.parent;
  if (p) st = p.name.slice(8);
  const run = (() => { let q = h.object; while (q && !q.name.startsWith('run_')) q = q.parent; return q ? q.name.slice(4) : ''; })();
  el.innerHTML = `${escapeHtml(h.object.userData.pick)} <span class="muted">${escapeHtml(st)}${run ? ' · ' + escapeHtml(run.split('__').slice(1).join('/') || 'export') : ''}</span>`;
  el.style.left = `${e.clientX}px`; el.style.top = `${e.clientY}px`;
  el.classList.remove('hidden');
  clearTimeout(el._t); el._t = setTimeout(() => el.classList.add('hidden'), 4000);
});

// resize + loop
function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
  S.dirty = true;
}
window.addEventListener('resize', () => { resize(); if (S.camPreset) applyPreset(S.camPreset); });
let last = performance.now();
function tick(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (S.playing && !scrubbing && S.views.length) {
    S.t += dt * S.speed;
    if (S.t > S.duration) { if (S.loop) S.t = 0; else { S.t = S.duration; setPlaying(false); } }
    applyTime();
  }
  if (controls.update()) S.dirty = true;
  if (S.dirty) { renderer.render(scene, camera); S.dirty = false; }
  requestAnimationFrame(tick);
}

// periodic index refresh: flag when the run on screen (or a room choice) has newer data
async function refreshIndex() {
  try {
    const idx = await fetchJSON(`${DATA}index.json`);
    const before = new Map(S.index.runs.map((r) => [r.id, r.mtime]));
    const shown = S.views.map((v) => v.man.id);
    let changed = shown.some((id) => { const r = idx.runs.find((x) => x.id === id); return !r || r.mtime !== before.get(id); });
    if (S.mode === 'room') {
      const oldChoice = Object.keys(S.layout.STATIONS).map((n) => (roomChoice(n) || {}).id);
      S.index = idx;
      const newChoice = Object.keys(S.layout.STATIONS).map((n) => (roomChoice(n) || {}).id);
      changed = changed || oldChoice.join() !== newChoice.join();
    } else S.index = idx;
    $('#reloadBtn').classList.toggle('hidden', !changed);
    if (!$('#runs').hidden) renderRunList();
  } catch (e) { /* offline: try again later */ }
}
$('#reloadBtn').onclick = () => { const keepT = S.t; setMode(S.mode, S.runId, false, S.camPreset).then(() => { S.t = keepT; applyTime(); }); };
window.addEventListener('hashchange', () => {
  const h = readHash();
  if (h.mode !== S.mode || h.id !== S.runId) setMode(h.mode, h.id, false, h.cam);
});

// debug hook for the headless check (tools/shoot.py)
Object.assign(window.__viewer, {
  THREE, scene, camera, controls, renderer,
  setTime(t) { S.t = t; applyTime(); renderer.render(scene, camera); },
  setPlaying, applyPreset,
  pose(runId, name) { const v = S.views.find((x) => x.id === runId); return v ? v.pose(name) : null; },
  runs() { return S.views.map((v) => ({ id: v.id, station: v.station, frame: v.frame, n: v.man.n_frames, stride: v.man.stride })); },
  render() { renderer.render(scene, camera); },
});

// ------------------------------------------------------------------ boot
(async function boot() {
  resize();
  showLoading(true, 'Loading the run index');
  try { S.index = await fetchJSON(`${DATA}index.json`); } catch (e) { showLoading(true, `No data yet (${e.message}). Run sync.py.`); return; }
  S.layout = S.index.layout;
  setupStudio(S.layout);
  const h = readHash();
  requestAnimationFrame(tick);
  await setMode(h.mode, h.id, false, h.cam);
  applyTime();
  setInterval(refreshIndex, 60000);
})();
