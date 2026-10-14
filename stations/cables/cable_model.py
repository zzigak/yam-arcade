"""Cable model for the cables station: rest shapes, a framed-rod spring network that gives a libuipc rod a curved
rest shape plus isotropic bending and torsion, and the libuipc hook that installs it.

Why not just FEM.Rod
--------------------
Genesis's `FEM.Rod` is libuipc `HookeanSpring` (stretch) + `KirchhoffRodBending` (KRB). KRB's hinge energy is
    E = (k pi r^4 / 4) |kappa|^2 / L0,   kappa = 2 e0 x e1 / (|e0||e1| + e0.e1)
with kappa from the CURRENT positions only; rest positions enter only through L0
(libuipc src/backends/cuda/finite_element/constitutions/kirchhoff_rod_bending.cu; its spec says "the energy penalizes
deviations from the straight configuration"). The rest shape is always straight, there is no rest-angle attribute,
and a pre-bent polyline springs straight (tests/rest_shape_test.py). The rod also has no twist.

Springs between the rod's own vertices cannot fix this properly: any energy of distances between points of a planar
curve changes only at second order when the curve bends out of its plane, so a curved rest shape held that way has
zero out-of-plane bending stiffness (and a 3-point hinge energy at a curved equilibrium has none either, pre-stressed
or not). The rod needs material frames.

The framed rod
--------------
Positions-only rods get material frames from ghost points (Umetani, Schmidt, Stam 2014, "Position-based elastic
rods"). Each edge j of the cable carries three ghost points around its midpoint, at distance h and 120 degrees apart
in the plane normal to the edge (directions from a parallel-transported frame). Edge + its ghosts form a rigid frame
(all 10 distances among the 5 points held by stiff springs). Three ghosts, not two: their centroid stays on the cable
axis however the frame turns, so gravity on the ghosts exerts no torque about the axis (with two ghosts it did, and a
curved cable lying on the table slowly rolled its frames, tests/rest_shape_test.py). Adjacent frames share the hinge
vertex x_j, and the 4 x 4 = 16 springs between {x_(j-1), ghosts_(j-1)} and {x_(j+1), ghosts_j} carry the hinge's
relative rotation. Their stiffnesses are fitted per hinge (non-negative least
squares) so that the hinge's rotational stiffness matrix is
    K_j = (EI / l) (I - t t^T) + (GJ / l) t t^T
i.e. isotropic bending EI and torsion GJ = EI / (1 + nu). All rest lengths are measured on the rest shape, so the rest
shape is stress-free and is what the cable returns to. The ghosts carry 5 % of the cable's mass, take part in no
contact (libuipc's no-collision contact element) and are not drawn: they discretise the orientation of the cable's
cross-section, they are not objects. All springs are libuipc SoftVertexStitch instances (pairwise, conservative);
KRB is left at a negligible value. Framed.effective_moduli() measures EI (two bending planes) and GJ on the whole
cable from the same energy the GPU solver minimises.
"""
import numpy as np
from scipy.optimize import lsq_linear

D_HAT = 1.0e-3  # IPC contact_d_hat used by the station
NU_JACKET = 0.45  # PVC / rubber jacket: GJ = EI / (1 + nu) for a solid round section (J = 2 I, G = E / 2(1 + nu))
H_FRAC = 1.5  # ghost offset h = H_FRAC * segment length (1.5: the 9 hinge springs then reach GJ to 2-3 %)
GHOST_MASS_FRAC = 0.05  # share of the cable mass carried by the ghost points
GHOST_RADIUS = 0.0005  # ghost polylines' nominal radius (they never collide)
FRAME_STIFF = 10.0  # frame springs = FRAME_STIFF x the largest hinge spring of the cable (rigid frames)
KAPPA_MAX = 20.0  # hinge spring stiffness bound, x the stiffness a straight hinge needs
KRB_E_TINY = 1.0  # Pa, per-vertex KRB stiffness left on cable and ghosts (EI ~ 1e-11 N m^2, negligible)


# ---------------------------------------------------------------------------------------------------------- shapes
def resample(curve, seg):
    """Points along a dense curve at equal arc length close to `seg`; returns (points, spacing)."""
    curve = np.asarray(curve, float)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))])
    n_seg = max(2, int(round(s[-1] / seg)))
    t = np.linspace(0.0, s[-1], n_seg + 1)
    return np.stack([np.interp(t, s, curve[:, k]) for k in range(3)], axis=1), s[-1] / n_seg


def _turtle(pieces, step=1e-4):
    """Dense curve from pieces: ('line', length), ('arc', radius, signed angle), ('helix', radius, signed angle, rise).
    Heading starts along +x in the xy plane."""
    p = np.zeros(3)
    h = 0.0
    out = [p.copy()]
    for pc in pieces:
        if pc[0] == "line":
            n = max(1, int(pc[1] / step))
            for _ in range(n):
                p = p + (pc[1] / n) * np.array([np.cos(h), np.sin(h), 0.0])
                out.append(p.copy())
        else:
            R, ang = pc[1], pc[2]
            rise = pc[3] if pc[0] == "helix" else 0.0
            n = max(2, int(abs(ang) * R / step))
            for _ in range(n):
                dh = ang / n
                hm = h + 0.5 * dh
                ds = R * abs(dh)
                p = p + np.array([ds * np.cos(hm), ds * np.sin(hm), rise / n])
                h += dh
                out.append(p.copy())
    return np.array(out)


def rest_curve(spec, center=True):
    """Dense rest centreline for a cable spec (local frame, z up, xy-centred, lowest point at z = 0). Shapes:
    coil   - flat coil from packaging: `turns` turns of radius R rising `pitch` per turn, straight tails
    ubend  - hairpin: two legs joined by a half circle of radius R (optional `skew` makes the legs unequal)
    scurve - two opposite arcs (R, angle) with straight ends
    wave   - a meander of `n_arcs` alternating arcs (R, angle)
    loop   - straight lead, one helical loop (R, rises `pitch`) so it crosses over itself, straight lead
    arc    - a single arc of `angle`
    pieces - an explicit list of _turtle pieces (arcs / helices / lines), length = their sum"""
    L, k = spec["length"], spec["shape"]
    if k == "pieces":
        pieces = spec["pieces"]
    elif k == "coil":
        R, turns, pitch = spec["R"], spec["turns"], spec["pitch"]
        tail = 0.5 * (L - 2 * np.pi * R * turns)
        pieces = [("line", tail), ("helix", R, 2 * np.pi * turns, pitch * turns), ("line", tail)]
    elif k == "ubend":
        R, sk = spec["R"], spec.get("skew", 0.0)
        leg = 0.5 * (L - np.pi * R)
        pieces = [("line", leg + sk), ("arc", R, np.pi), ("line", leg - sk)]
    elif k == "scurve":
        R, a = spec["R"], spec["angle"]
        end = 0.5 * (L - 2 * R * a)
        pieces = [("line", end), ("arc", R, a), ("arc", R, -a), ("line", end)]
    elif k == "wave":
        R, a, n = spec["R"], spec["angle"], spec["n_arcs"]
        end = 0.5 * (L - n * R * a)
        pieces = [("line", end), ("arc", R, a / 2)]
        pieces += [("arc", R, -a if i % 2 == 0 else a) for i in range(n - 1)]
        pieces += [("arc", R, (a / 2) * (1 if (n - 1) % 2 == 0 else -1)), ("line", end)]
    elif k == "loop":
        R, pitch = spec["R"], spec["pitch"]
        lead = 0.5 * (L - 2 * np.pi * R)
        pieces = [("line", lead), ("helix", R, 2 * np.pi, pitch), ("line", lead)]
    elif k == "arc":
        pieces = [("arc", L / spec["angle"], spec["angle"])]
    else:
        raise ValueError(k)
    c = _turtle(pieces)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    c = c[s <= L + 1e-9]
    if center:
        c[:, :2] -= 0.5 * (c[:, :2].min(0) + c[:, :2].max(0))
        c[:, 2] -= c[:, 2].min()
    return c


def pieces_length(pieces):
    """Arc length of a _turtle piece list."""
    L = 0.0
    for pc in pieces:
        if pc[0] == "line":
            L += pc[1]
        else:
            rise = pc[3] if pc[0] == "helix" else 0.0
            L += np.hypot(abs(pc[2]) * pc[1], rise)
    return L


def rot(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def place(points, xy, yaw_deg, z_bottom, tilt=None, flip=False, extra=None, return_transform=False):
    """Rigidly place a local polyline: optional flip (180 deg about x), optional (axis, deg) tilt, yaw about z; then
    the lowest point of (points + extra) to z_bottom and the xy box centre to xy. With return_transform: (p, R, t)
    with p = points @ R.T + t."""
    p = np.asarray(points, float)
    allp = p if extra is None else np.vstack([p, extra])
    R = np.eye(3)
    if flip:
        R = rot([1, 0, 0], np.pi) @ R
    if tilt is not None:
        R = rot(tilt[0], np.radians(tilt[1])) @ R
    R = rot([0, 0, 1], np.radians(yaw_deg)) @ R
    q = allp @ R.T
    t = np.zeros(3)
    t[:2] = np.asarray(xy, float) - 0.5 * (q[:, :2].min(0) + q[:, :2].max(0))
    t[2] = z_bottom - q[:, 2].min()
    out = p @ R.T + t
    return (out, R, t) if return_transform else out


# ------------------------------------------------------------------------------------------------ hinge geometry
def turning_angles(P):
    """Turning angle (rad, >= 0) at each interior vertex 1..n-2."""
    e = np.diff(P, axis=0)
    a, b = e[:-1], e[1:]
    c = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.arccos(np.clip(c, -1.0, 1.0))


def bishop_frames(P):
    """Unit tangents and parallel-transported (u, v) per edge, u_0 as vertical as possible."""
    e = np.diff(P, axis=0)
    t = e / np.linalg.norm(e, axis=1, keepdims=True)
    ref = np.array([0.0, 0.0, 1.0]) if abs(t[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u0 = ref - ref.dot(t[0]) * t[0]
    u = [u0 / np.linalg.norm(u0)]
    for j in range(1, len(t)):
        uj = _align(t[j - 1], t[j]) @ u[-1]
        uj = uj - uj.dot(t[j]) * t[j]
        u.append(uj / np.linalg.norm(uj))
    u = np.array(u)
    return t, u, np.cross(t, u)


def _align(a, b):
    """Minimal rotation taking direction a to direction b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    ax = np.cross(a, b)
    s = np.linalg.norm(ax)
    if s < 1e-12:
        if a.dot(b) > 0:
            return np.eye(3)
        return rot(np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0]), np.pi)
    return rot(ax, np.arctan2(s, a.dot(b)))


def _sym6(M):
    s = np.sqrt(2.0)
    return np.array([M[0, 0], M[1, 1], M[2, 2], s * M[0, 1], s * M[0, 2], s * M[1, 2]])


# --------------------------------------------------------------------------------------------- framed-rod design
N_GHOST = 3


class Framed:
    """Framed-rod discretisation of one cable whose rest (= initial) centreline is P. Vertex index space: cable
    0..n-1, then ghost family k (k = 0..N_GHOST-1, one ghost per edge) at n + k (n - 1) + j. Springs (i, j) with
    rest length, kappa (N/m) and family (0 frame, 1 hinge) in that index space."""

    def __init__(self, P, radius, EI, GJ=None, h_frac=H_FRAC, frame_stiff=FRAME_STIFF, hinge_scale=None):
        P = np.asarray(P, float)
        self.P, self.n, self.radius, self.EI = P, len(P), radius, EI
        self.GJ = EI / (1.0 + NU_JACKET) if GJ is None else GJ
        le = np.linalg.norm(np.diff(P, axis=0), axis=1)
        self.cable_rest = le
        self.seg = le.mean()
        self.h = h_frac * self.seg
        t, u, v = bishop_frames(P)
        self.t, self.u, self.v = t, u, v
        mid = 0.5 * (P[1:] + P[:-1])
        ang = 2 * np.pi * np.arange(N_GHOST) / N_GHOST
        self.G = [mid + self.h * (np.cos(a) * u + np.sin(a) * v) for a in ang]
        self.X0 = np.vstack([P] + self.G)
        n = self.n
        gi = lambda k, j: n + k * (n - 1) + j  # noqa: E731
        self.gi = gi
        springs, fam, fit_res = [], [], []
        kmax_h = np.zeros(n)  # stiffest hinge spring per hinge vertex
        w_fit = np.ones(6)
        for j in range(1, n - 1):  # hinge at x_j between edges j-1 and j
            A = [j - 1] + [gi(k, j - 1) for k in range(N_GHOST)]
            B = [j + 1] + [gi(k, j) for k in range(N_GHOST)]
            tb = t[j - 1] + t[j]
            tb /= np.linalg.norm(tb)
            lb = 0.5 * (le[j - 1] + le[j])
            hs = 1.0 if hinge_scale is None else hinge_scale[j - 1]  # e.g. the section moulded into a plug
            Kt = hs * ((self.EI / lb) * (np.eye(3) - np.outer(tb, tb)) + (self.GJ / lb) * np.outer(tb, tb))
            cols, prs = [], []
            for a in A:
                for b in B:
                    pa, pb = self.X0[a], self.X0[b]
                    uh = (pb - pa) / np.linalg.norm(pb - pa)
                    J = np.cross(pb - P[j], uh)  # d|pb - pa| / d omega, frame B rotating about x_j
                    cols.append(_sym6(np.outer(J, J)))
                    prs.append((a, b))
            # bounded least squares; the bound (KAPPA_MAX x the stiffness a straight hinge needs) keeps near-degenerate
            # spring directions from taking absurd stiffnesses
            k_ref = hs * (self.EI / lb) * (self.seg ** 2 + self.h ** 2) / (self.h * self.seg) ** 2
            Am, bv = np.array(cols).T * w_fit[:, None], _sym6(Kt) * w_fit
            kap = lsq_linear(Am, bv, bounds=(0.0, KAPPA_MAX * k_ref), method="bvls").x
            fit_res.append(np.linalg.norm(Am @ kap - bv) / np.linalg.norm(bv))
            kmax_h[j] = kap.max()
            for (a, b), k in zip(prs, kap):
                if k > 1e-6 * kap.max():
                    springs.append((a, b, np.linalg.norm(self.X0[a] - self.X0[b]), k))
                    fam.append(1)
        # frame springs: FRAME_STIFF x the stiffest hinge spring next to the edge (stiff plug sections get stiff frames)
        kmax_h[0], kmax_h[-1] = kmax_h[1], kmax_h[-2]
        ref = np.median(kmax_h[1:-1])
        self.kappa_frame = frame_stiff * kmax_h.max() if hinge_scale is None else frame_stiff * ref
        for j in range(n - 1):  # rigid frame: each ghost to both edge ends, and ghost-ghost
            kf = frame_stiff * (kmax_h.max() if hinge_scale is None else max(kmax_h[j], kmax_h[j + 1], ref))
            pairs = [(e, gi(k, j)) for k in range(N_GHOST) for e in (j, j + 1)]
            pairs += [(gi(a, j), gi(b, j)) for a in range(N_GHOST) for b in range(a + 1, N_GHOST)]
            for a, b in pairs:
                springs.append((a, b, np.linalg.norm(self.X0[a] - self.X0[b]), kf))
                fam.append(0)
        self.springs = np.array([s[:2] for s in springs], np.int64)
        self.rest = np.array([s[2] for s in springs])
        self.kappa = np.array([s[3] for s in springs])
        self.family = np.array(fam)
        self.fit_residual = float(np.max(fit_res))

    def energy(self, X, stretch_ks=None):
        """Network energy of configuration X ((3n-2, 3), numpy or torch); the same terms the GPU solver gets (without
        its dt^2 scaling). stretch_ks adds the cable's own stretch springs as a penalty."""
        import torch
        T = torch.as_tensor(X, dtype=torch.float64)
        pr = torch.as_tensor(self.springs)
        d = (T[pr[:, 0]] - T[pr[:, 1]]).norm(dim=1)
        E = (0.5 * torch.as_tensor(self.kappa) * (d - torch.as_tensor(self.rest)) ** 2).sum()
        if stretch_ks is not None:
            dc = (T[1:self.n] - T[:self.n - 1]).norm(dim=1)
            E = E + 0.5 * stretch_ks * ((dc - torch.as_tensor(self.cable_rest)) ** 2).sum()
        return E if isinstance(X, torch.Tensor) else float(E)

    def bent(self, dtheta, axis="b"):
        """Whole configuration with every hinge rotated by dtheta about a frame-attached axis ('b': v, 'n': u, the two
        bending directions; 't': tangent, twist); edge frames (ghosts) move rigidly with their edges."""
        n, e = self.n, np.diff(self.P, axis=0)
        Q = [np.eye(3)]
        for j in range(1, n - 1):
            a = {"b": self.v[j - 1], "n": self.u[j - 1], "t": self.t[j - 1]}[axis]
            Q.append(Q[-1] @ rot(a, dtheta))
        X = [self.P[0]]
        for j in range(n - 1):
            X.append(X[-1] + Q[j] @ e[j])
        X = np.array(X)
        out = np.empty_like(self.X0)
        out[:n] = X
        for j in range(n - 1):
            m0, m1 = 0.5 * (self.P[j] + self.P[j + 1]), 0.5 * (X[j] + X[j + 1])
            for k in range(N_GHOST):
                out[self.gi(k, j)] = m1 + Q[j] @ (self.G[k][j] - m0)
        return out

    def effective_moduli(self, dkappa=0.5):
        """EI about the two frame axes and GJ measured on the whole cable: a uniform curvature (or twist-rate) change
        dkappa (1/m) applied hinge by hinge, central second difference of the network energy."""
        out = {}
        L = (self.n - 2) * self.seg
        E0 = self.energy(self.X0)
        for ax, name in (("b", "EI_1"), ("n", "EI_2"), ("t", "GJ")):
            dth = dkappa * self.seg
            out[name] = (self.energy(self.bent(dth, ax)) + self.energy(self.bent(-dth, ax)) - 2 * E0) / dkappa ** 2 / L
        return out


def relaxed_bending_EI(fr, dkappa=0.5, axis="b", iters=400):
    """EI with the cable centreline held in a uniformly bent shape and the ghosts free to settle (frame springs are
    finite, frames may twist): the bending rigidity the GPU model actually has. Torch L-BFGS over ghost positions."""
    import torch
    X = torch.tensor(fr.bent(dkappa * fr.seg, axis), dtype=torch.float64)
    cab = X[:fr.n].clone()
    gh = X[fr.n:].clone().requires_grad_(True)
    opt = torch.optim.LBFGS([gh], max_iter=iters, tolerance_grad=1e-14, tolerance_change=1e-18, history_size=50,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        E = fr.energy(torch.cat([cab, gh]))
        E.backward()
        return E

    for _ in range(5):
        opt.step(closure)
    E = float(fr.energy(torch.cat([cab, gh]).detach()))
    return 2 * E / dkappa ** 2 / ((fr.n - 2) * fr.seg)


# ----------------------------------------------------------------------------------------------- libuipc hookup
_HOOKS = []


def install_finalize_hook():
    """Run every registered hook(coupler) right before libuipc's world.init (IPCCoupler._finalize_ipc)."""
    from genesis.engine.couplers.ipc_coupler.coupler import IPCCoupler
    if IPCCoupler.__dict__.get("_yam_cables_hooked"):
        return
    orig = IPCCoupler._finalize_ipc

    def _finalize(self):
        for h in _HOOKS:
            h(self)
        orig(self)

    IPCCoupler._finalize_ipc = _finalize
    IPCCoupler._yam_cables_hooked = True


def add_hook(fn):
    install_finalize_hook()
    _HOOKS.append(fn)


def clear_hooks():
    _HOOKS.clear()


def find_rod_slot(coupler, entity):
    from uipc.backend import SceneVisitor
    from uipc.geometry import SimplicialComplexSlot
    idx = list(coupler.fem_solver.entities).index(entity)
    for slot in SceneVisitor(coupler._ipc_scene).geometries():
        if not isinstance(slot, SimplicialComplexSlot):
            continue
        meta = slot.geometry().meta()
        st = meta.find("solver_type")
        if st is None or st.view()[0] != "rod":
            continue
        if int(meta.find("entity_idx").view()[0]) == idx:
            return slot
    raise RuntimeError("rod geometry not found in the IPC scene")


_SVS = {}


def add_framed_cable(scene, gs, fr, rho_line, friction, color, krb_only=False):
    """Add a framed cable to a Genesis scene: the cable rod (FEM.Rod, contact on) and, unless krb_only, its N_GHOST
    ghost polylines plus the spring network (installed at build by a finalize hook). rho_line: kg/m of the real cable.
    krb_only: plain FEM.Rod with KRB carrying the full EI (straight rest shape), for comparison.
    Returns dict(cable=entity, ghosts=[...], info={...} filled at build)."""
    r = fr.radius
    L = fr.seg * (fr.n - 1)
    gfrac = 0.0 if krb_only else GHOST_MASS_FRAC
    I = np.pi * r ** 4 / 4
    cable = scene.add_entity(
        gs.morphs.Polyline(points=[tuple(p) for p in fr.P], radius=r),
        material=gs.materials.FEM.Rod(E=4e7, bending_stiffness=(fr.EI / I) if krb_only else KRB_E_TINY,
                                      rho=(1.0 - gfrac) * rho_line / (np.pi * r ** 2), friction_mu=friction),
        surface=gs.surfaces.Default(color=tuple(color)),
    )
    out = dict(cable=cable, ghosts=[], info={})
    if krb_only:
        return out
    m_ghost_rod = gfrac * rho_line * L / N_GHOST
    for G in fr.G:
        lg = np.linalg.norm(np.diff(G, axis=0), axis=1).sum()
        out["ghosts"].append(scene.add_entity(
            gs.morphs.Polyline(points=[tuple(p) for p in G], radius=GHOST_RADIUS),
            material=gs.materials.FEM.Rod(E=1e4, bending_stiffness=KRB_E_TINY,
                                          rho=m_ghost_rod / (np.pi * GHOST_RADIUS ** 2 * lg), friction_mu=0.0),
            surface=gs.surfaces.Default(color=tuple(color)),
        ))
    info = out["info"]

    def hook(coupler):
        import uipc
        from uipc.constitution import SoftVertexStitch
        slots = [find_rod_slot(coupler, e) for e in [cable] + out["ghosts"]]
        n = fr.n
        offs = np.array([0] + [n + k * (n - 1) for k in range(N_GHOST)])
        for s, X in zip(slots, [fr.P] + fr.G):
            pos = s.geometry().positions().view().squeeze(-1)
            err = float(np.abs(pos - X).max())
            assert err < 1e-6, f"rod vertices differ from the design by {err}"
            bs = uipc.view(s.geometry().vertices().find("bending_stiffness"))
            bs[:] = KRB_E_TINY
        for s in slots[1:]:  # ghosts take part in no contact
            coupler._ipc_no_collision_contact.apply_to(s.geometry())
        key = id(coupler)
        if key not in _SVS:
            _SVS[key] = SoftVertexStitch()
            coupler._ipc_constitution_tabular.insert(_SVS[key])
        which = np.searchsorted(offs, fr.springs, side="right") - 1  # slot of each endpoint
        local = fr.springs - offs[which]
        cnt = 0
        for sa in range(len(slots)):
            for sb in range(len(slots)):
                m = (which[:, 0] == sa) & (which[:, 1] == sb)
                if not m.any():
                    continue
                g = _SVS[key].create_geometry((slots[sa], slots[sb]), local[m].astype(np.int32), kappa=1.0,
                                              rest_length=1.0)
                kv = uipc.view(g.instances().find("kappa"))
                kv[:] = fr.kappa[m].reshape(kv.shape)
                rl = uipc.view(g.instances().find("rest_length"))
                rl[:] = fr.rest[m].reshape(rl.shape)
                coupler._ipc_objects.create(f"cable{cable.idx}_frame_{sa}{sb}").geometries().create(g)
                cnt += int(m.sum())
        info.update(n_springs=cnt, kappa_hinge_max=float(fr.kappa[fr.family == 1].max()),
                    kappa_frame=float(fr.kappa_frame))

    add_hook(hook)
    return out
