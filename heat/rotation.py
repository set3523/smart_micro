"""Auto-split from python314_mesh_ver5_mesh2.py"""
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

def calculate_rotation_center(comm, xdmf_path):
    """
    회전 중심을 계산합니다.
    우선순위 1: create_mesh_from_geo에서 생성한 'rotation_center.txt' 파일
    우선순위 2: Mass(1925) 영역의 중심 (Glass가 없으므로 Mass를 기준)
    우선순위 3: 전체 메쉬의 중심
    """
    # 1. 파일에서 읽기 시도 (가장 정확함)
    if os.path.exists("rotation_center.txt"):
        try:
            center = np.loadtxt("rotation_center.txt")
            if comm.rank == 0:
                print(f"[Utils] Loaded rotation center from 'rotation_center.txt': {center}")
            return center
        except:
            pass

    if comm.rank == 0:
        print(f"[Utils] Calculating rotation center from mesh (Fallback)...")

    with io.XDMFFile(comm, xdmf_path, "r") as xdmf:
        domain = xdmf.read_mesh(name="mesh")
        cell_tags = xdmf.read_meshtags(domain, name="PhysicalGroups")

    # 2. Try Mass (1925) - Glass가 없으므로 Mass가 가장 믿을만함
    id_mass = 1925
    target_cells = cell_tags.find(id_mass)

    rotation_center = None

    if len(target_cells) > 0:
        domain.topology.create_connectivity(domain.topology.dim, 0)
        vertex_indices = mesh.entities_to_geometry(domain, domain.topology.dim, target_cells, True)
        unique_vertices = np.unique(vertex_indices)
        coords = domain.geometry.x[unique_vertices]

        min_pt = np.min(coords, axis=0)
        max_pt = np.max(coords, axis=0)
        center = (min_pt + max_pt) / 2.0
        rotation_center = center[:3]

        if comm.rank == 0:
            print(f"[Utils] Calculated Rotation Center from Mass: {rotation_center}")
    else:
        # 3. Fallback to entire mesh center
        coords = domain.geometry.x
        min_pt = np.min(coords, axis=0)
        max_pt = np.max(coords, axis=0)
        center = (min_pt + max_pt) / 2.0
        rotation_center = center[:3]
        if comm.rank == 0:
            print(
                f"[Utils] \033[93mWarning: Mass domain not found. Using Mesh Bounding Box Center: {rotation_center}\033[0m")

    return rotation_center
