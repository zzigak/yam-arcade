# cables / newton: status (2026-10-13, final)

**Done: the export and the checks pass** (`checks.json`, all pass; `DONE` written).

Whole run: drop, settle, then 2 cables sorted by the YAM Ultra = **65.3 s simulated in 148.7 s wall** (+4.8 s build).
That is 2.2-2.4 s of wall time per simulated second in every phase. The GPU was shared with 2 other processes at 99 %
utilisation. The time includes per-frame readback and rendering at 1280x720.

| phase | wall per simulated s |
|---|---|
| drop + settle | 2.19 |
| robot phases | 2.3 |

Genesis + libuipc for comparison: ~55-230 s/s for drop + settle, and 600-830 s/s while grasping.

- **Budget:** SolverVBD at 60 fps x 40 substeps x 3 iterations. With fewer than ~36 substeps per frame the thin cables
  spike: millimetre interpenetration and no settling.
- **Settled pile:** KE 9e-8 J, 116 crossings, 5 interlocked pairs.
- **Physics:**
  - stretch [0.986, 1.036]
  - max overlap 0.76 mm (pad on cable under grip), 0.42 mm between cables
  - rest-shape retention alone 0.98-0.998
  - a straight-rest control opens to 0.12
- **Sorting:**
  - barrel_arc goes into the blue bin with the right arm.
  - cat6_scurve goes into the orange bin with the left arm.
  - Both are 100 % inside.
  - Plug slip 0.2 / 0.7 mm; grip 6.0 N per finger.

Details are in `stations/cables_newton/NOTES.md`.
