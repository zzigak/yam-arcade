"""Two patches to Genesis' legacy SPH <-> rigid coupling, applied at runtime by this station only.

1. Union per link (a physics fix).  Upstream couples every particle to every coupled geom independently.  That is
   right for a link made of ONE geom, but a hollow container has to be built from many convex pieces, and then a
   particle near the wall sees 2-5 overlapping pieces within the SPH support radius and gets the boundary-pressure
   push (and the container gets the reaction force) 2-5 times over.  Measured effect at rest: wall particles kicked to
   2-6 m/s and the water "boiling" at 0.12-0.16 m/s mean speed forever.  The fix applies the boundary-pressure term
   once per link, from the nearest piece (the union of the pieces, whose signed distance is the minimum over them).
   The velocity response (remove the inward relative velocity) still runs against every nearby piece in turn, as
   upstream: that sequential projection is what keeps particles out of concave floor/wall corners, and coplanar
   neighbours find nothing left to remove, so it is not double counted.

2. Performance (results unchanged).
`LegacyCoupler._func_collide_with_rigid_geom_robust` evaluates the SDF *gradient* of every coupled geom for every
particle each substep, before checking whether the particle is anywhere near that geom.  This station couples the
water to ~700 small convex pieces (the hollow containers), so ~13 M gradient evaluations per substep dominate the run
time.  The gradient is only used when the particle is within the coupling band (influence > 0.1, i.e. signed distance
< coup_softness * ln 10) or within the SPH support radius (the boundary-pressure term).  This version evaluates the
signed distance first and the gradient only in that case; the physics is the same code path as upstream otherwise
(copied from genesis/engine/couplers/legacy_coupler.py, Genesis 1.3.1 branch ipc_rod_and_mpm).

Call `apply(scene)` after creating the scene and before `scene.build()`.
"""
import quadrants as qd

import genesis as gs
import genesis.utils.sdf as sdf
from genesis.engine.couplers.legacy_coupler import LegacyCoupler
from genesis.utils import array_class
from genesis.utils.geom import qd_transform_by_trans_quat


@qd.data_oriented
class _LinkBounds:
    """Per-link bounding sphere of the coupled geometry, in the link frame, padded by the coupling reach.
    A particle outside it cannot be within the SPH support radius / coupling band of any of the link's geoms, so the
    patched kernels skip that link's per-geom SDF queries (pure early-out: results unchanged)."""

    def __init__(self):
        self.center = None
        self.radius = None

    def alloc(self, scene):
        """Before scene.build(): fields sized to the rigid links / geoms, radius 'infinite' (no skipping) until
        setup()."""
        n = scene.sim.rigid_solver.n_links
        ng = sum(len(link.geoms) for link in scene.sim.rigid_solver.links)
        self.center = qd.Vector.field(3, dtype=gs.qd_float, shape=n)
        self.radius = qd.field(dtype=gs.qd_float, shape=n)
        self.radius.fill(1e9)
        self.gcenter = qd.Vector.field(3, dtype=gs.qd_float, shape=max(ng, 1))
        self.gradius = qd.field(dtype=gs.qd_float, shape=max(ng, 1))
        self.gradius.fill(1e9)

    def setup(self, scene, reach):
        import numpy as np
        from genesis.utils.misc import tensor_to_array
        rs = scene.sim.rigid_solver
        n = rs.n_links
        centers = np.zeros((n, 3), dtype=np.float32)
        radii = np.full(n, 1e9, dtype=np.float32)
        ng = self.gradius.shape[0]
        gcenters = np.zeros((ng, 3), dtype=np.float32)
        gradii = np.full(ng, 1e9, dtype=np.float32)
        for i, link in enumerate(rs.links):
            geoms = list(link.geoms)
            if not geoms or any(g.type == gs.GEOM_TYPE.PLANE for g in geoms):
                continue     # no geometry (radius irrelevant) or an infinite plane: always check
            p = tensor_to_array(link.get_pos()).reshape(3)
            q = tensor_to_array(link.get_quat()).reshape(4)
            w, x, y, z = q
            R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            Ls = [(tensor_to_array(g.get_verts()).reshape(-1, 3) - p) @ R for g in geoms]
            L = np.concatenate(Ls)
            c = 0.5 * (L.min(0) + L.max(0))
            centers[i] = c
            radii[i] = np.linalg.norm(L - c, axis=1).max() + reach
            for g, Lg in zip(geoms, Ls):
                cg = 0.5 * (Lg.min(0) + Lg.max(0))
                gcenters[g.idx] = cg
                gradii[g.idx] = np.linalg.norm(Lg - cg, axis=1).max() + reach
        assert self.center is not None and self.center.shape[0] == n, "call alloc_link_bounds before scene.build"
        self.center.from_numpy(centers)
        self.radius.from_numpy(radii)
        self.gcenter.from_numpy(gcenters)
        self.gradius.from_numpy(gradii)
        return centers, radii


LINK_BOUNDS = _LinkBounds()


def alloc_link_bounds(scene):
    LINK_BOUNDS.alloc(scene)


def setup_link_bounds(scene, reach=0.012):
    """Call once after scene.build() and before the first step (reach > SPH support radius 8 mm and the 4.6 mm
    coupling band)."""
    import os
    if os.environ.get("WATER_NO_LINK_BOUNDS"):
        return None
    return LINK_BOUNDS.setup(scene, reach)


@qd.data_oriented
class UnionSPHCoupler(LegacyCoupler):
    def build(self) -> None:
        """Upstream build, but the per-(particle, geom) SPH normal history - which the patched kernel does not use -
        is allocated with a single geom slot instead of n_geoms (1 GB and two full copies per substep with ~1900
        convex pieces; see BoundarySPHSolver._kernel_reorder_particles)."""
        n_geoms = self.rigid_solver.n_geoms_
        self.rigid_solver.n_geoms_ = 1
        try:
            super().build()
        finally:
            self.rigid_solver.n_geoms_ = n_geoms

    def reset(self, envs_idx=None) -> None:
        """Upstream's per-env reset kernel loops over every geom of the (now single-slot) normal history."""
        assert not (self._rigid_mpm and self.mpm_solver.enable_CPIC), "station has no MPM"
        if self._rigid_sph:
            self.sph_rigid_normal.fill(0)

    @qd.func
    def _func_collide_with_rigid_geom_robust(
        self,
        pos_world,
        vel,
        mass,
        pressure,
        normal_prev,
        geom_idx,
        batch_idx,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        links_info: array_class.LinksInfo,
        rigid_info: array_class.RigidInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        signed_dist = sdf.sdf_func_world(geom_idx, batch_idx, pos_world, geoms_state, geoms_info, sdf_info)
        influence = qd.min(qd.exp(-signed_dist / max(1e-10, geoms_info.coup_softness[geom_idx])), 1)
        normal_rigid = normal_prev
        if influence > 0.1 or signed_dist < self.sph_solver._support_radius:
            normal_rigid = sdf.sdf_func_normal_world(
                geom_idx, batch_idx, pos_world, geoms_state, geoms_info, rigid_info, sdf_info, collider_static_config
            )
            if influence > 0.1:
                vel = self._func_collide_in_rigid_geom(
                    pos_world, vel, mass, normal_rigid, influence, geom_idx, batch_idx, geoms_info, links_state, rigid_info
                )
            link_idx = geoms_info.link_idx[geom_idx]
            I_l = [link_idx, batch_idx] if qd.static(self.rigid_solver._options.batch_links_info) else link_idx
            if signed_dist < self.sph_solver._support_radius and pressure > 0 and not links_info.is_fixed[I_l]:
                pressure_force = (
                    -2.0
                    * pressure
                    * self.sph_solver._particle_volume
                    * self.sph_solver.cubic_kernel_plane_integral(signed_dist)
                    * normal_rigid
                )
                self.rigid_solver._func_apply_coupling_force(link_idx, batch_idx, pos_world, pressure_force, links_state)
                vel = vel - pressure_force * (rigid_info.substep_dt[None] / mass)
        return vel, normal_rigid


    @qd.kernel
    def sph_rigid(
        self,
        f: qd.i32,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        links_info: array_class.LinksInfo,
        rigid_info: array_class.RigidInfo,
        sdf_info: array_class.SDFInfo,
        collider_static_config: qd.template(),
    ):
        for i_p, i_b in qd.ndrange(self.sph_solver._n_particles, self.sph_solver._B):
            if self.sph_solver.particles_ng_reordered[i_p, i_b].active:
                pos = self.sph_solver.particles_reordered[i_p, i_b].pos
                mass = self.sph_solver.particles_info_reordered[i_p, i_b].mass
                for i_l in range(self.rigid_solver.n_links):
                    I_l = [i_l, i_b] if qd.static(self.rigid_solver._options.batch_links_info) else i_l
                    best_g = -1
                    best_sd = gs.qd_float(1e9)
                    c = qd_transform_by_trans_quat(LINK_BOUNDS.center[i_l], links_state.pos[i_l, i_b],
                                                   links_state.quat[i_l, i_b])
                    near = (pos - c).norm() < LINK_BOUNDS.radius[i_l]
                    for i_g in range(links_info.geom_start[I_l], links_info.geom_end[I_l]):
                        near_g = False
                        if near:
                            cg = qd_transform_by_trans_quat(LINK_BOUNDS.gcenter[i_g], links_state.pos[i_l, i_b],
                                                            links_state.quat[i_l, i_b])
                            near_g = (pos - cg).norm() < LINK_BOUNDS.gradius[i_g]
                        if near_g and geoms_info.needs_coup[i_g]:
                            sd = sdf.sdf_func_world(i_g, i_b, pos, geoms_state, geoms_info, sdf_info)
                            if sd < best_sd:
                                best_sd = sd
                                best_g = i_g
                            # velocity response against EVERY nearby piece (sequential projection: a particle in a
                            # concave floor/wall corner must not move into either surface; coplanar neighbours
                            # find nothing left to remove, so nothing is double counted)
                            influence = qd.min(qd.exp(-sd / max(1e-10, geoms_info.coup_softness[i_g])), 1)
                            if influence > 0.1:
                                normal = sdf.sdf_func_normal_world(
                                    i_g, i_b, pos, geoms_state, geoms_info, rigid_info, sdf_info,
                                    collider_static_config
                                )
                                self.sph_solver.particles_reordered[i_p, i_b].vel = self._func_collide_in_rigid_geom(
                                    pos, self.sph_solver.particles_reordered[i_p, i_b].vel, mass, normal, influence,
                                    i_g, i_b, geoms_info, links_state, rigid_info
                                )
                    # boundary-pressure push once per link, from its nearest piece (upstream counts it per piece)
                    pressure = self.sph_solver.particles_reordered[i_p, i_b].p
                    if best_g >= 0 and best_sd < self.sph_solver._support_radius and pressure > 0 \
                            and not links_info.is_fixed[I_l]:
                        normal = sdf.sdf_func_normal_world(
                            best_g, i_b, pos, geoms_state, geoms_info, rigid_info, sdf_info, collider_static_config
                        )
                        pressure_force = (
                            -2.0
                            * pressure
                            * self.sph_solver._particle_volume
                            * self.sph_solver.cubic_kernel_plane_integral(best_sd)
                            * normal
                        )
                        self.rigid_solver._func_apply_coupling_force(i_l, i_b, pos, pressure_force, links_state)
                        self.sph_solver.particles_reordered[i_p, i_b].vel = (
                            self.sph_solver.particles_reordered[i_p, i_b].vel
                            - pressure_force * (rigid_info.substep_dt[None] / mass)
                        )


def apply(scene):
    """Swap the scene's legacy coupler for the patched subclass. Call after gs.Scene(...) and before scene.build()."""
    coupler = scene.sim._coupler
    assert type(coupler) is LegacyCoupler, f"expected the legacy coupler, got {type(coupler)}"
    coupler.__class__ = UnionSPHCoupler


# ---------------------------------------------------------------------------------------------------------------
# 3. Boundary density (a physics fix, optional).  Genesis' SPH has no boundary particles: a particle next to a rigid
#    wall misses the wall side of its kernel, its density is under-estimated, the WCSPH clamp (rho >= rho0) then gives
#    it zero pressure, and the layer behind it is accelerated into the wall by the spurious pressure drop (measured:
#    second-layer particles at 1-3 m/s toward the wall in a resting pitcher).  This adds the wall's share of the kernel
#    analytically, rho0 * Phi(d), with d the union-SDF distance of the nearest piece of each coupled link and Phi the
#    cubic spline's mass beyond a plane at distance d (Phi(0) = 1/2; the integral of Genesis'
#    cubic_kernel_plane_integral).  With the density right, wall particles carry the right pressure and the coupler's
#    mirrored boundary-pressure term (which assumes it) balances them.
from genesis.engine.solvers.sph_solver import SPHSolver  # noqa: E402


@qd.data_oriented
class BoundarySPHSolver(SPHSolver):
    _bd_enabled = True

    @qd.kernel
    def _kernel_reorder_particles(self, f: qd.i32):
        self.sh.compute_reordered_idx(
            self._n_particles, self.particles.pos, self.particles_ng.active, self.particles_ng.reordered_idx
        )
        self.particles_ng_reordered.active.fill(False)
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                reordered_idx = self.particles_ng[i_p, i_b].reordered_idx
                self.particles_reordered[reordered_idx, i_b] = self.particles[i_p, i_b]
                self.particles_info_reordered[reordered_idx, i_b] = self.particles_info[i_p]
                self.particles_ng_reordered[reordered_idx, i_b].active = self.particles_ng[i_p, i_b].active

    @qd.kernel
    def _kernel_copy_from_reordered(self, f: qd.i32):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng[i_p, i_b].active:
                self.particles[i_p, i_b] = self.particles_reordered[self.particles_ng[i_p, i_b].reordered_idx, i_b]

    @qd.func
    def _boundary_volume_fraction(self, dist):
        """Phi(d) = int_d^h sigma(s) ds for the cubic spline (h = support radius); Phi(-d) = 1 - Phi(d)."""
        h = self._support_radius
        q = qd.min(qd.abs(dist) / h, 1.0)
        G = gs.qd_float(0.0)  # 16 * int_q^1 g(t) dt, with g the plane integral in units of 16/h
        if q <= 0.5:
            G1q = 0.0875 * q - 0.2 * q ** 6 + 0.3 * q ** 5 - q ** 3 / 6.0
            G = 16.0 * (0.0291666667 - G1q + 0.0333333333 - 0.03125)
        else:
            G2q = 0.1 * q - q ** 3 / 3.0 + 0.5 * q ** 4 - 0.3 * q ** 5 + q ** 6 / 15.0
            G = 16.0 * (0.0333333333 - G2q)
        res = G
        if dist < 0:
            res = 1.0 - G
        return res

    @qd.kernel
    def _kernel_add_boundary_rho(
        self,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        links_state: array_class.LinksState,
        links_info: array_class.LinksInfo,
        sdf_info: array_class.SDFInfo,
    ):
        for i_p, i_b in qd.ndrange(self._n_particles, self._B):
            if self.particles_ng_reordered[i_p, i_b].active:
                phi = gs.qd_float(0.0)
                for i_l in range(self._bd_n_links):
                    I_l = [i_l, i_b] if qd.static(self._bd_batch_links_info) else i_l
                    best = gs.qd_float(1e9)
                    pos = self.particles_reordered[i_p, i_b].pos
                    c = qd_transform_by_trans_quat(LINK_BOUNDS.center[i_l], links_state.pos[i_l, i_b],
                                                   links_state.quat[i_l, i_b])
                    near = (pos - c).norm() < LINK_BOUNDS.radius[i_l]
                    for i_g in range(links_info.geom_start[I_l], links_info.geom_end[I_l]):
                        near_g = False
                        if near:
                            cg = qd_transform_by_trans_quat(LINK_BOUNDS.gcenter[i_g], links_state.pos[i_l, i_b],
                                                            links_state.quat[i_l, i_b])
                            near_g = (pos - cg).norm() < LINK_BOUNDS.gradius[i_g]
                        if near_g and geoms_info.needs_coup[i_g]:
                            sd = sdf.sdf_func_world(i_g, i_b, pos, geoms_state, geoms_info, sdf_info)
                            best = qd.min(best, sd)
                    if best < self._support_radius:
                        phi += self._boundary_volume_fraction(best)
                self.particles_reordered[i_p, i_b].rho += phi * self.particles_info_reordered[i_p, i_b].rho

    def build(self):
        super().build()
        rs = self._sim.rigid_solver
        self._bd_n_links = rs.n_links
        self._bd_batch_links_info = rs._options.batch_links_info

    def substep_pre_coupling(self, f):
        assert self._pressure_solver == "WCSPH"
        if self.is_active:
            rs = self._sim.rigid_solver
            self._kernel_reorder_particles(f)
            self._kernel_compute_rho(f)
            if self._bd_enabled:
                self._kernel_add_boundary_rho(rs.dyn_state.geoms, rs.dyn_info.geoms, rs.dyn_state.links, rs.dyn_info.links,
                                              rs.collider._sdf._sdf_info)
            self._kernel_compute_non_pressure_forces(f, self._sim.cur_t)
            self._kernel_compute_pressure_forces(f)
            self._kernel_advect_velocity(f)


def apply_boundary_density(scene, enabled=True):
    """Swap the scene's SPH solver for BoundarySPHSolver (required with UnionSPHCoupler, whose normal history it no
    longer copies). Call after gs.Scene(...) and before scene.build()."""
    solver = scene.sim.sph_solver
    assert type(solver) is SPHSolver, f"expected the stock SPH solver, got {type(solver)}"
    solver.__class__ = BoundarySPHSolver
    solver._bd_enabled = enabled
