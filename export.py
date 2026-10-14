"""Station -> compositor export (see SPEC.md, "Export format").

Every station writes out/<station>/ with this Recorder; the Blender compositor reads only these files, so a station
never has to know anything about rendering.

    rec = Recorder("out/cables", station="cables", fps=30)
    rec.add_rigid("bin_red", "assets/bin.obj", material=dict(kind="plastic", color=(0.8, 0.1, 0.1)))
    rec.add_rod("cable_0", radius=0.004, material=dict(kind="rubber", color=(0.9, 0.2, 0.1)))
    rec.add_particles("water", radius=0.0025, material=dict(kind="water"))
    rec.add_meshseq("water_surface", material=dict(kind="water"))
    for each recorded frame:
        rec.frame(t, rigid={"bin_red": (pos, quat_wxyz)}, rods={"cable_0": pts}, particles={"water": pts},
                  meshes={"water_surface": (V, F)})
    rec.close()

All positions are in the station-local frame (layout.py), metres, z up. Quaternions are (w, x, y, z) like Genesis.
A rigid's mesh file is expressed in the frame whose pose is recorded (for Genesis entities: the link frame).
Material kinds the compositor knows: wood, paint, plastic, rubber, ceramic, glass, metal, water, fabric, paper,
emissive; 'color' is sRGB 0-1; optional 'roughness', 'texture' (image path, for meshes without embedded textures).
"""
import json
import os
import shutil

import numpy as np


class Recorder:
    def __init__(self, out_dir, station, fps):
        self.out_dir = os.path.abspath(out_dir)
        os.makedirs(os.path.join(self.out_dir, "meshes"), exist_ok=True)
        self.station, self.fps = station, fps
        self.objects = {}
        self.t = []
        self.series = {}
        self.n_meshseq = {}

    def _mesh(self, mesh):
        """Copy a mesh (and an OBJ's .mtl + textures, if beside it) into out/<station>/meshes/; return relative path."""
        mesh = os.path.abspath(mesh)
        dst = os.path.join(self.out_dir, "meshes", os.path.basename(mesh))
        if mesh != dst:
            shutil.copy2(mesh, dst)
            if mesh.endswith(".obj"):
                src_dir = os.path.dirname(mesh)
                for line in open(mesh, errors="ignore"):
                    if line.startswith("mtllib"):
                        mtl = os.path.join(src_dir, line.split(None, 1)[1].strip())
                        if os.path.exists(mtl):
                            shutil.copy2(mtl, os.path.join(self.out_dir, "meshes", os.path.basename(mtl)))
                            for l2 in open(mtl, errors="ignore"):
                                if l2.split() and l2.split()[0].startswith("map_"):
                                    tex = os.path.join(src_dir, l2.split()[-1])
                                    if os.path.exists(tex):
                                        shutil.copy2(tex, os.path.join(self.out_dir, "meshes", os.path.basename(tex)))
        return os.path.relpath(dst, self.out_dir)

    def _add(self, name, **kw):
        assert name not in self.objects, f"duplicate object {name}"
        mat = kw.get("material")
        if isinstance(mat, dict) and mat.get("texture") and os.path.isabs(str(mat["texture"])):
            tex = str(mat["texture"])  # copy an absolute texture next to the meshes and point at the copy
            dst = os.path.join(self.out_dir, "meshes", os.path.basename(tex))
            if os.path.abspath(tex) != dst:
                shutil.copy2(tex, dst)
            kw["material"] = dict(mat, texture=os.path.relpath(dst, self.out_dir))
        self.objects[name] = kw

    def add_rigid(self, name, mesh, material, static_pose=None, scale=1.0):
        """static_pose=(pos, quat_wxyz) for things that never move (recorded once, no per-frame data needed)."""
        kw = dict(kind="rigid", mesh=self._mesh(mesh), material=material, scale=scale)
        if static_pose is not None:
            kw["static_pose"] = [list(map(float, static_pose[0])), list(map(float, static_pose[1]))]
        self._add(name, **kw)

    def add_rod(self, name, radius, material):
        self._add(name, kind="rod", radius=float(radius), material=material)

    def add_particles(self, name, radius, material):
        self._add(name, kind="particles", radius=float(radius), material=material)

    def add_meshseq(self, name, material):
        """A per-frame triangle mesh (water surface, FEM body), written to meshseq/<name>/NNNNN.npz (V float32, F int32)."""
        self._add(name, kind="meshseq", material=material, dir=f"meshseq/{name}")
        os.makedirs(os.path.join(self.out_dir, "meshseq", name), exist_ok=True)

    def frame(self, t, rigid=None, rods=None, particles=None, meshes=None):
        i = len(self.t)
        self.t.append(float(t))
        for name, (pos, quat) in (rigid or {}).items():
            self.series.setdefault(f"{name}/pos", []).append(np.array(pos, np.float32).reshape(3))
            self.series.setdefault(f"{name}/quat", []).append(np.array(quat, np.float32).reshape(4))
        for name, pts in list((rods or {}).items()) + list((particles or {}).items()):
            # np.array copies: np.asarray would keep a view, i.e. the caller's whole source array alive per frame
            # (a 993-frame particle export held one full copy of its particle history per frame and hit 117 GB)
            self.series.setdefault(f"{name}/pts", []).append(np.array(pts, np.float32).reshape(-1, 3))
        for name, (V, F) in (meshes or {}).items():
            np.savez_compressed(os.path.join(self.out_dir, "meshseq", name, f"{i:05d}.npz"),
                                V=np.asarray(V, np.float32), F=np.asarray(F, np.int32))
        for k, v in self.series.items():
            assert len(v) == i + 1, f"{k} missing at frame {i}: every dynamic object must be recorded every frame"

    def close(self, extra_meta=None):
        np.savez_compressed(os.path.join(self.out_dir, "traj.npz"), t=np.array(self.t, np.float32),
                            **{k: np.stack(v) for k, v in self.series.items()})
        meta = dict(station=self.station, fps=self.fps, n_frames=len(self.t), objects=self.objects)
        meta.update(extra_meta or {})
        json.dump(meta, open(os.path.join(self.out_dir, "meta.json"), "w"), indent=1)
