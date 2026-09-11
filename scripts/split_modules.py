"""Split python314_mesh_ver5_mesh2.py into microwave_sim package modules."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "python314_mesh_ver5_mesh2.py"
OUT = ROOT / "microwave_sim"

MODULE_HEADER = '''"""Auto-split from python314_mesh_ver5_mesh2.py"""
import csv
import gmsh
import h5py
import numpy as np
import os
import random
import sys
import ufl
from dolfinx import fem, mesh, io, plot, geometry
from dolfinx.io import gmsh as dolfinx_gmsh
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx.fem.petsc

from microwave_sim.constants import (
    freq, omega, c0, mu0, eps0, k0, DEFAULT_REF_FILE, get_ref_file, set_ref_file,
)

'''

SPLITS = {
    "mesh/geo.py": [(27, 180), (182, 197)],
    "em/geo_solver.py": [(203, 541)],
    "heat/transient.py": [
        (544, 843), (846, 974), (977, 1147), (1149, 1320),
        (1322, 1543), (2864, 3125), (3127, 3399), (3403, 3687),
    ],
    "heat/uniformity.py": [(1546, 1714), (1717, 1921), (1924, 2087)],
    "heat/rotation.py": [(2093, 2147)],
    "optimization/schedule.py": [
        (2150, 2316), (2319, 2559), (2561, 2722), (2726, 2861),
        (3690, 3819), (3821, 4032), (4035, 4232),
    ],
}

lines = SRC.read_text(encoding="utf-8").splitlines(keepends=True)

for rel_path, ranges in SPLITS.items():
    out_path = OUT / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for start, end in ranges:
        chunks.extend(lines[start - 1 : end])
    content = MODULE_HEADER + "".join(chunks)
    # Replace module-level ref_file usage
    content = content.replace(
        'ref_file = "cal_by_gmsh3/microwave_geometry_and_stackdumpling_0.xdmf"',
        "# ref_file moved to microwave_sim.constants",
    )
    out_path.write_text(content, encoding="utf-8")
    print(f"Wrote {out_path} ({len(chunks)} lines)")

print("Done.")
