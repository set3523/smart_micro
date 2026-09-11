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

def solve_electromagnetic_problem(comm, domain, cell_tags, facet_tags, marker_map, result_file="result/solution_geo.xdmf"):
    """
    전자기장 해석을 수행합니다. (Weak Form Port)
    """
    print("[Solver] Setting up function spaces and materials...")

    id_air = marker_map.get("air")
    if id_air is None:
        id_air = marker_map.get("Background_Air")

    id_port = marker_map.get("port")
    id_wall = marker_map.get("wall")
    id_glass = marker_map.get("glass")
    ids_mass = [val for key, val in marker_map.items() if key.startswith("mass")]

    if id_port is None:
        print("Warning: 'port' physical group not found in .geo file!")
    
    print(f"[Solver] Material IDs found: Air={id_air}, Glass={id_glass}, Mass={ids_mass}")
    print(f"[Solver] Boundary IDs found: Port={id_port}, Wall={id_wall}")

    degree = 2
    V = fem.functionspace(domain, ("N1curl", degree))

    Q_space = fem.functionspace(domain, ("DG", 0))
    eps_r = fem.Function(Q_space, dtype=np.complex128)
    eps_r.x.array[:] = 1.0 + 0.0j

    def set_material(tag_ids, val):
        if not tag_ids: return
        if isinstance(tag_ids, int): tag_ids = [tag_ids]
        for tag in tag_ids:
            cells = cell_tags.find(tag)
            if len(cells) > 0:
                eps_r.x.array[cells] = val

    if id_air:
        set_material(id_air, 1.0 + 0.0j)
    if id_glass:
        set_material(id_glass, 5.5 + 0.0j)
    set_material(ids_mass, 45.0 - 15.0j)

    mu_r = 1.0
    k0_val = 2 * np.pi * 2.45e9 / 3e8

    bcs = []
    if id_wall is not None:
        facets_wall = facet_tags.find(id_wall)
        dofs_wall = fem.locate_dofs_topological(V, domain.topology.dim - 1, facets_wall)
        zero_f = fem.Function(V)
        zero_f.x.array[:] = 0.0
        bc_wall = fem.dirichletbc(zero_f, dofs_wall)
        bcs.append(bc_wall)

    E = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    n = ufl.FacetNormal(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    a = (1 / mu_r) * ufl.inner(ufl.curl(E), ufl.curl(v)) * ufl.dx \
        - k0_val ** 2 * ufl.inner(eps_r * E, v) * ufl.dx
    L = ufl.inner(fem.Constant(domain, np.array([0, 0, 0], dtype=np.complex128)), v) * ufl.dx

    if id_port is not None:
        E_inc = fem.Function(V)
        E_inc.x.array[:] = 1.0 + 0.0j
        
        # Impedance boundary condition term (LHS)
        a += 1j * k0_val * ufl.inner(ufl.cross(n, E), ufl.cross(n, v)) * ds(id_port)
        # Excitation term (RHS)
        L += 2 * 1j * k0_val * ufl.inner(ufl.cross(n, E_inc), ufl.cross(n, v)) * ds(id_port)

    print("[Solver] Solving linear system...")
    E_h = fem.Function(V)
    E_h.name = "E_field"

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }
    problem = fem.petsc.LinearProblem(a, L, bcs=bcs, u=E_h, petsc_options=petsc_options, petsc_options_prefix="pc")
    problem.solve()

    print("[Solver] Post-processing results...")
    V_Q = fem.functionspace(domain, ("DG", 0))
    Q = fem.Function(V_Q)
    Q.name = "HeatSource"
    Q_expr_ufl = 0.5 * omega * eps0 * (-ufl.imag(eps_r)) * ufl.real(ufl.dot(E_h, ufl.conj(E_h)))
    Q_expr = fem.Expression(Q_expr_ufl, V_Q.element.interpolation_points)
    Q.interpolate(Q_expr)

    target_power = 700.0
    dx_sub = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    total_power_sim = 0.0
    try:
        power_integrals = [fem.assemble_scalar(fem.form(Q * dx_sub(tag))) for tag in ids_mass]
        total_power_sim = comm.allreduce(sum(power_integrals), op=MPI.SUM)
    except Exception as e:
        print(f"[Solver] Power integration failed: {e}.")

    if isinstance(total_power_sim, complex):
        total_power_sim = total_power_sim.real

    if total_power_sim > 1e-12:
        scale_factor = target_power / total_power_sim
        print(f"[Solver] \033[96mScaling Q by {scale_factor:.2f} to match target power {target_power} W\033[0m")
        Q.x.array[:] *= scale_factor
        E_h.x.array[:] *= np.sqrt(scale_factor)
    else:
        print("[Solver] \033[91mWarning: Absorbed power is zero. Cannot scale.\033[0m")

    V_lagrange = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    E_h_lagrange = fem.Function(V_lagrange)
    E_h_lagrange.name = "E_field_Lagrange"
    E_h_expr = fem.Expression(E_h, V_lagrange.element.interpolation_points)
    E_h_lagrange.interpolate(E_h_expr)

    with io.XDMFFile(comm, result_file, "w") as xdmf:
        xdmf.write_mesh(domain)
        cell_tags.name = "PhysicalGroups"
        xdmf.write_meshtags(cell_tags, domain.geometry)
        xdmf.write_function(E_h_lagrange)
        xdmf.write_function(Q)

    print(f"[Solver] Done. Results saved to {result_file}")


def solve_electromagnetic_problem2(comm, domain, cell_tags, facet_tags, marker_map,
                                   result_file="result/solution_geo.xdmf"):
    """
    전자기장 해석을 수행하고 포트 데이터(S11 등)와 포인팅 벡터를 저장합니다.
    """
    print("[Solver] Setting up function spaces and materials...")

    id_air = marker_map.get("air")
    if id_air is None: id_air = marker_map.get("Background_Air")
    id_port = marker_map.get("port")
    id_wall = marker_map.get("wall")
    id_glass = marker_map.get("glass")
    ids_mass = [val for key, val in marker_map.items() if key.startswith("mass")]

    if id_port is None:
        print("Warning: 'port' physical group not found in .geo file!")

    degree = 2
    V = fem.functionspace(domain, ("N1curl", degree))

    Q_space = fem.functionspace(domain, ("DG", 0))
    eps_r = fem.Function(Q_space, dtype=np.complex128)
    eps_r.x.array[:] = 1.0 + 0.0j

    def set_material(tag_ids, val):
        if not tag_ids: return
        if isinstance(tag_ids, int): tag_ids = [tag_ids]
        for tag in tag_ids:
            cells = cell_tags.find(tag)
            if len(cells) > 0:
                eps_r.x.array[cells] = val

    if id_air: set_material(id_air, 1.0 + 0.0j)
    if id_glass: set_material(id_glass, 5.5 + 0.0j)
    set_material(ids_mass, 45.0 - 15.0j)

    mu_r = 1.0
    k0_val = 2 * np.pi * 2.45e9 / 3e8

    bcs = []
    if id_wall is not None:
        facets_wall = facet_tags.find(id_wall)
        dofs_wall = fem.locate_dofs_topological(V, domain.topology.dim - 1, facets_wall)
        zero_f = fem.Function(V)
        zero_f.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(zero_f, dofs_wall))

    E = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    n = ufl.FacetNormal(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    a = (1 / mu_r) * ufl.inner(ufl.curl(E), ufl.curl(v)) * ufl.dx \
        - k0_val ** 2 * ufl.inner(eps_r * E, v) * ufl.dx
    L = ufl.inner(fem.Constant(domain, np.array([0, 0, 0], dtype=np.complex128)), v) * ufl.dx

    E_inc = fem.Function(V)
    if id_port is not None:
        E_inc.x.array[:] = 1.0 + 0.0j
        a += 1j * k0_val * ufl.inner(ufl.cross(n, E), ufl.cross(n, v)) * ds(id_port)
        L += 2 * 1j * k0_val * ufl.inner(ufl.cross(n, E_inc), ufl.cross(n, v)) * ds(id_port)

    print("[Solver] Solving linear system...")
    E_h = fem.Function(V)
    E_h.name = "E_field"

    petsc_options = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    problem = fem.petsc.LinearProblem(a, L, bcs=bcs, u=E_h, petsc_options=petsc_options, petsc_options_prefix="pc_em")
    problem.solve()

    # -------------------------------------------------------------------------
    # [추가됨] H-field 및 Poynting Vector 계산 및 저장 준비
    # -------------------------------------------------------------------------
    print("[Solver] Calculating H-field and Poynting Vector...")

    # 벡터 저장을 위한 Lagrange 공간 (시각화용)
    V_vec = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

    # 1. H-field (자기장)
    H_h = fem.Function(V_vec)
    H_h.name = "H_field"
    # H = (-1 / j*w*mu) * curl(E)
    H_ufl = (-1.0 / (1j * omega * mu0)) * ufl.curl(E_h)
    H_expr = fem.Expression(H_ufl, V_vec.element.interpolation_points)
    H_h.interpolate(H_expr)

    # 2. Poynting Vector (에너지 흐름)
    S_h = fem.Function(V_vec)
    S_h.name = "Poynting_Vector"
    # S_avg = 0.5 * Re(E x H*)
    S_ufl = 0.5 * ufl.real(ufl.cross(E_h, ufl.conj(H_h)))
    S_expr = fem.Expression(S_ufl, V_vec.element.interpolation_points)
    S_h.interpolate(S_expr)

    # -------------------------------------------------------------------------
    # S11 및 Port 파라미터 계산
    # -------------------------------------------------------------------------
    print("[Solver] Calculating Port Parameters (S11)...")

    # S11 계산을 위한 Flux (S_ufl 사용)
    flux_term = ufl.dot(S_ufl, n)

    P_net_complex = 0.0 + 0.0j
    S11_dB = 0.0
    E_vec_avg = np.zeros(3, dtype=np.complex128)

    if id_port:
        try:
            # Port를 통과하는 순수 전력 (Net Power)
            P_flux = fem.assemble_scalar(fem.form(flux_term * ds(id_port)))
            P_net_real = comm.allreduce(P_flux, op=MPI.SUM)

            # 들어가는 방향을 양수로 변환 (보통 n은 밖을 향함)
            P_in_net = -P_net_real

            # 입사 전력 추정 (Analytical)
            Z0 = 377.0
            E_inc_tan_sq = ufl.inner(E_inc, E_inc)
            P_inc_form = fem.form((1.0 / (2.0 * Z0)) * E_inc_tan_sq * ds(id_port))
            P_inc = comm.allreduce(fem.assemble_scalar(P_inc_form), op=MPI.SUM)

            # 반사 전력 계산: P_net = P_inc - P_ref  => P_ref = P_inc - P_net
            P_ref = P_inc - P_in_net

            S11_abs = 0.0
            if P_inc > 1e-12:
                # S11 = sqrt(P_ref / P_inc)
                ratio = P_ref / P_inc
                if ratio < 0: ratio = 0  # 수치 오차 방지
                S11_abs = np.sqrt(ratio)

            S11_dB = 20 * np.log10(S11_abs + 1e-16)

            # 복소수 리턴값 유지를 위해 (호환성)
            P_net_complex = -P_in_net + 0j

            if comm.rank == 0:
                print(f"[Solver] Port Power Net In: {P_in_net:.4e} W")
                print(f"[Solver] Port Power Incident (Est): {P_inc:.4e} W")
                print(f"[Solver] Port Power Reflected (Est): {P_ref:.4e} W")
                print(f"[Solver] S11: {S11_dB:.2f} dB")

            # Calculate average E-field vector at port
            area_form = fem.form(fem.Constant(domain, PETSc.ScalarType(1.0)) * ds(id_port))
            port_area = comm.allreduce(fem.assemble_scalar(area_form), op=MPI.SUM)
            if port_area > 1e-9:
                for i in range(3):
                    e_comp_form = fem.form(E_h[i] * ds(id_port))
                    val = comm.allreduce(fem.assemble_scalar(e_comp_form), op=MPI.SUM)
                    E_vec_avg[i] = val / port_area

            if comm.rank == 0:
                print(f"[Solver] Port E-field Vector (Avg): {E_vec_avg}")

        except Exception as e:
            if comm.rank == 0: print(f"[Solver] Warning: S11 calc failed: {e}")

    # -------------------------------------------------------------------------
    # Heat Source (Q) 계산 및 스케일링
    # -------------------------------------------------------------------------
    V_Q = fem.functionspace(domain, ("DG", 0))
    Q = fem.Function(V_Q)
    Q.name = "HeatSource"
    Q_expr_ufl = 0.5 * omega * eps0 * (-ufl.imag(eps_r)) * ufl.real(ufl.dot(E_h, ufl.conj(E_h)))
    Q_expr = fem.Expression(Q_expr_ufl, V_Q.element.interpolation_points)
    Q.interpolate(Q_expr)

    target_power = 700.0
    dx_sub = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    total_power_sim = 0.0
    try:
        power_integrals = [fem.assemble_scalar(fem.form(Q * dx_sub(tag))) for tag in ids_mass]
        total_power_sim = comm.allreduce(sum(power_integrals), op=MPI.SUM)
    except:
        pass

    if isinstance(total_power_sim, complex): total_power_sim = total_power_sim.real

    if total_power_sim > 1e-12:
        scale_factor = target_power / total_power_sim
        Q.x.array[:] *= scale_factor
        # E, H, S 필드도 에너지 비율에 맞춰 스케일링 (시각화 정확도를 위해)
        # Power ~ E^2 이므로 E, H는 sqrt(scale) 배, S는 scale 배
        sqrt_scale = np.sqrt(scale_factor)
        E_h.x.array[:] *= sqrt_scale
        H_h.x.array[:] *= sqrt_scale
        S_h.x.array[:] *= scale_factor

    # -------------------------------------------------------------------------
    # 결과 저장 (XDMF) - Poynting Vector 포함
    # -------------------------------------------------------------------------
    # E-field 시각화용 (Lagrange)
    V_lagrange = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    E_h_lagrange = fem.Function(V_lagrange)
    E_h_lagrange.name = "E_field_Lagrange"
    E_h_expr = fem.Expression(E_h, V_lagrange.element.interpolation_points)
    E_h_lagrange.interpolate(E_h_expr)

    with io.XDMFFile(comm, result_file, "w") as xdmf:
        xdmf.write_mesh(domain)
        cell_tags.name = "PhysicalGroups"
        xdmf.write_meshtags(cell_tags, domain.geometry)

        # 저장할 필드들
        xdmf.write_function(Q)
        xdmf.write_function(E_h_lagrange)  # 전기장
        xdmf.write_function(H_h)  # 자기장 (추가됨)
        xdmf.write_function(S_h)  # 포인팅 벡터 (추가됨)

    return P_net_complex, S11_dB, E_vec_avg
