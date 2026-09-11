"""
DOLFINx-based EM solver for Microwave Oven Simulation.
이 스크립트는 시뮬레이션 로직을 함수화하여
다른 파이썬 스크립트에서 import하여 사용할 수 있도록 합니다.
"""

import petsc4py.PETSc
from dolfinx import fem, mesh, io
import dolfinx.fem.petsc
from dolfinx.mesh import create_box, CellType, refine

# 병렬 계산을 위한 MPI
from mpi4py import MPI

# 방정식을 정의하기 위한 UFL
import ufl

# NumPy
import numpy as np

# 경로 관리를 위한 os
import os
import time

import resource
from mpi4py import MPI

def print_peak_memory():
    comm = MPI.COMM_WORLD
    # ru_maxrss는 KB 단위로 리턴됩니다.
    peak_mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mem_gb = peak_mem_kb / (1024 * 1024)  # KB -> GB 변환

    # 모든 프로세스 중 가장 높게 찍힌 피크값 합산 (전체 노드 부하 계산용)
    total_peak_gb = comm.reduce(peak_mem_gb, op=MPI.SUM, root=0)
    # 단일 프로세스 중 최댓값 (특정 코어 폭발 확인용)
    max_single_peak_gb = comm.reduce(peak_mem_gb, op=MPI.MAX, root=0)

    if comm.rank == 0:
        print(f"\n[Peak Memory Report]")
        print(f" - 전체 코어 합산 피크 메모리: {total_peak_gb:.2f} GB")
        print(f" - 단일 프로세스 최대 피크: {max_single_peak_gb:.2f} GB")
        print("-" * 30)

def run_microwave_simulation(
        number="checkcheck",
        n_elem_x=93,
        n_elem_y=84,
        n_elem_z=93,
        output_dir="result",
        comm=MPI.COMM_WORLD,
        PortLA = 1,PortLB = 1,
        PortTA = 1,PortTB = 1,
        PortBA = 1,PortBB = 1,
        degree = 1,
        petsc_options_em = {
        "ksp_error_if_not_converged": True,
        "ksp_type": "gmres",
        "pc_type": "gamg",
        "ksp_rtol": 1e-8,  # 1e-8
        "ksp_atol": 1e-12,
        "ksp_max_it": 1000,
        "ksp_view": None,
        "ksp_monitor": None,  # 수렴 과정 출력 (유지)
        "ksp_gmres_restart": 200,  # GMRES 재시작 빈도 (유지)
        "mg_levels_ksp_type" : "chebyshev",
        "mg_levels_pc_type" : "jacobi",
        "pc_gamg_coarse_eq_dof" : 3,
        "pc_gamg_symm" : 'false',
        "pc_gamg_cycle_type" : "w",
        "pc_gamg_threshold" : 0.1,
        },
        cell_hex_tet = "hex",
    petsc_options_prefix_em="test",
    ):

    """
    전자레인지 EM 시뮬레이션을 실행하고 결과를 파일로 저장합니다.
    (Weak Form Port Implementation)
    """

    # === 0. MPI 커뮤니케이터 설정 ===
    rank = comm.rank
    if rank == 0:
        pprint.pprint(petsc_options_em)
    # === 1. 설정 및 메쉬 생성 ===
    if rank == 0:
        print(f"[Sim Func - Step 1] Creating uniform mesh with create_box... ({n_elem_x}x{n_elem_y}x{n_elem_z})")

    # GMSH 파일의 좌표계(단위: m)와 일치시킵니다.
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([0.31, 0.26, 0.31])
    cell_type_selected = None
    if(cell_hex_tet == "hex"):
        cell_type_selected = CellType.hexahedron
    else:
        cell_type_selected = CellType.tetrahedron

    domain = create_box(
        comm,  # MPI.COMM_WORLD 대신 인자로 받은 comm 사용
        [p0, p1],  # 두 모서리 좌표 (단위: m)
        [n_elem_x, n_elem_y, n_elem_z],  # 함수 인자로 받은 해상도 사용
        cell_type=cell_type_selected
    )

    # 메쉬 정제
    msh = domain  # 코드의 나머지 부분에서 'domain' 변수를 사용합니다.

    # === 2. 물리 상수 및 재료 물성 정의 ===
    if rank == 0:
        print("[Sim Func - Step 2] Defining physical constants and material properties...")

    # Gmsh에서 부여한 '이름표(Physical Group)' 번호 (이제는 논리적 마커로 사용)
    pork_id, air_id = 1, 2
    pec_id, portt_id = 3, 6  # (pec_id는 이 코드에서 명시적으로 사용되진 않음)
    portl_id, portb_id = 5, 4  # (논리적 마커)

    # 물리 상수 (단위: m, s, H, F ...)
    #freq = 2.45e9   # 전자레인지 주파수 (Hz)
    freq = 2.45e9 / 5  # 전자레인지 주파수 (Hz)
    omega = 2 * np.pi * freq
    c0 = 299792458.0  # 진공 중 빛의 속도 (m/s)
    mu0 = 4 * np.pi * 1e-7  # 진공 중 투자율 (H/m)
    eps0 = 1 / (mu0 * c0 ** 2)  # 진공 중 유전율 (F/m)
    k0 = omega / c0  # 진공 중 파수 (rad/m)

    # 재료 물성 (상대값)
    eps_r_pork_re = 49.0
    tand_pork = 0.3
    eps_r_pork = eps_r_pork_re * (1 - 1j * tand_pork)  # 복소 유전율
    mu_r_pork = 1.0 + 0.0j
    eps_r_air = 1.0 + 0.0j
    mu_r_air = 1.0 + 0.0j

    # test 이거 끄면 Q가 0으로 뜸
    #eps_r_air = eps_r_pork

    if rank == 0:
        print("[Sim Func - Step 2] Finished definitions.")

    # === 3. 함수 공간 정의 ===
    if rank == 0:
        print("[Sim Func - Step 3] Defining function spaces...")


    V = fem.functionspace(domain, ("N1curl", degree))
    V_mat = fem.functionspace(domain, ("DG", 0))

    # === 4. 재료 물성 함수 설정 (좌표 기반) ===
    if rank == 0:
        print("[Sim Func - Step 4] Setting up material property functions...")

    dx = 0.31 / n_elem_x
    dy = 0.26 / n_elem_y
    dz = 0.31 / n_elem_z
    # 셀 중심에서 가장 먼 꼭짓점까지의 거리 (Conservative Selection용 여유분)
    mesh_tol = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) / 2.0

    def pork_region(x):  # uppersum
        # === 설정 파라미터 ===
        base_radius = 0.04  # 원래 의도한 반지름

        # [보정] 메쉬 오차만큼 반지름을 키움 (구, 정사면체 공용)
        eff_radius = base_radius + mesh_tol

        # 높이 설정
        y_center_bottom = 0.08  # 기존 물체들(구, 정사면체)의 높이
        y_center_top = 0.18  # 새로 추가할 구들의 높이

        # 정육면체 설정
        cube_side = 0.09  # 두 큐브 사이 유격 0.01(1cm)을 위한 변의 길이
        cube_half = cube_side / 2.0

        # [보정] 메쉬 오차만큼 큐브 범위를 키움
        eff_cube_half = cube_half + mesh_tol

        y_center_cube = 0.19  # 정육면체 중심 높이

        # ---------------------------------------------------------
        # 1. 구 (Sphere) 영역 정의
        # ---------------------------------------------------------
        sphere_configs = [
            # [기존] 높이 0.08인 구들
            (0.105, y_center_bottom, 0.105),
            (0.205, y_center_bottom, 0.205),

            # [추가] 높이 0.18인 구들 (1mm 유격 적용 좌표)
            # 0.200(오른쪽 구) - 0.08(지름합) - 0.001(유격) = 0.119
            (0.105, y_center_top, 0.119),
            (0.105, y_center_top, 0.200)
        ]

        in_spheres = np.zeros_like(x[0], dtype=bool)
        for cx, cy, cz in sphere_configs:
            dist_sq = (x[0] - cx) ** 2 + (x[1] - cy) ** 2 + (x[2] - cz) ** 2
            # [적용] eff_radius 사용
            in_spheres = np.logical_or(in_spheres, dist_sq <= (eff_radius ** 2))

        # ---------------------------------------------------------
        # 2. 정사면체 (Tetrahedron) 영역 정의
        #    (eff_radius를 사용하여 꼭짓점을 정의하면 자동으로 커짐)
        # ---------------------------------------------------------

        # 정사면체 기하학 상수 계산 (확장된 반지름 기준)
        H = 2.0 * eff_radius
        r_base = H / np.sqrt(2)

        # [정사면체 꼭짓점 정의]
        v_top = np.array([0.0, eff_radius, 0.0])
        v_b1 = np.array([r_base, -eff_radius, 0.0])
        v_b2 = np.array([r_base * np.cos(2 * np.pi / 3), -eff_radius, r_base * np.sin(2 * np.pi / 3)])
        v_b3 = np.array([r_base * np.cos(4 * np.pi / 3), -eff_radius, r_base * np.sin(4 * np.pi / 3)])

        verts_up = [v_top, v_b1, v_b2, v_b3]
        verts_down = [v * np.array([1.0, -1.0, 1.0]) for v in verts_up]

        # --- 내부 판별 함수 (Convex Hull 방식) ---
        def is_inside_convex(points, vertices):
            faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
            n_points = points.shape[1]
            inside = np.ones(n_points, dtype=bool)
            centroid = np.mean(vertices, axis=0)

            for face in faces:
                p0 = vertices[face[0]]
                p1 = vertices[face[1]]
                p2 = vertices[face[2]]
                vec1 = p1 - p0
                vec2 = p2 - p0
                normal = np.cross(vec1, vec2)
                norm_len = np.linalg.norm(normal)
                if norm_len == 0: continue
                normal = normal / norm_len
                if np.dot(normal, centroid - p0) > 0:
                    normal = -normal
                diff = points - p0[:, np.newaxis]
                dot_product = np.einsum('ij, i -> j', diff, normal)
                inside = np.logical_and(inside, dot_product <= 0)
            return inside

        # (A) 바닥에 서 있는 정사면체
        cx1, cz1 = 0.105, 0.205
        local_points_1 = np.array([x[0] - cx1, x[1] - y_center_bottom, x[2] - cz1])
        in_tetra_up = is_inside_convex(local_points_1, np.array(verts_up))

        # (B) 거꾸로 서 있는 정사면체
        cx2, cz2 = 0.205, 0.105
        local_points_2 = np.array([x[0] - cx2, x[1] - y_center_bottom, x[2] - cz2])
        in_tetra_down = is_inside_convex(local_points_2, np.array(verts_down))

        # ---------------------------------------------------------
        # 3. 정육면체 (Cube) 영역 정의
        # ---------------------------------------------------------
        cube_centers = [
            (0.205, 0.105),
            (0.205, 0.205)
        ]

        in_cubes = np.zeros_like(x[0], dtype=bool)

        for cx, cz in cube_centers:
            # [적용] eff_cube_half를 사용하여 범위 체크
            in_x = np.logical_and(x[0] >= cx - eff_cube_half, x[0] <= cx + eff_cube_half)
            in_y = np.logical_and(x[1] >= y_center_cube - eff_cube_half, x[1] <= y_center_cube + eff_cube_half)
            in_z = np.logical_and(x[2] >= cz - eff_cube_half, x[2] <= cz + eff_cube_half)

            in_current_cube = np.logical_and(in_x, np.logical_and(in_y, in_z))
            in_cubes = np.logical_or(in_cubes, in_current_cube)

        # ---------------------------------------------------------
        # 4. 최종 결과 합치기
        # ---------------------------------------------------------
        return np.logical_or(in_spheres, np.logical_or(np.logical_or(in_tetra_up, in_tetra_down), in_cubes))

    # --- 'pork'와 'air' 셀 찾기 ---
    tdim = domain.topology.dim
    all_cells_idx = np.arange(domain.topology.index_map(tdim).size_local + domain.topology.index_map(tdim).num_ghosts,
                              dtype=np.int32)

    # 'pork'에 해당하는 셀들의 인덱스 배열
    pork_cells = mesh.locate_entities(domain, tdim, pork_region)

    # 'air' 셀은 'pork'가 아닌 모든 셀
    air_cells_mask = np.ones(len(all_cells_idx), dtype=bool)
    if pork_cells.size > 0:
        air_cells_mask[pork_cells] = False
    air_cells = all_cells_idx[air_cells_mask]

    # 상대 유전율(epsilon_r) 함수 생성
    eps_r = fem.Function(V_mat)
    eps_r.name = "epsilon_r"
    eps_r.x.array[air_cells] = eps_r_air
    eps_r.x.array[pork_cells] = eps_r_pork

    # 상대 투자율(mu_r) 함수 생성
    mu_r = fem.Function(V_mat)
    mu_r.name = "mu_r"
    mu_r.x.array[:] = 1.0 + 0.0j #

    # (XDMF 저장을 위해 'subdomains' MeshTags 객체를 수동으로 생성)
    markers = np.full(len(all_cells_idx), air_id, dtype=np.int32)
    markers[pork_cells] = pork_id
    subdomains = mesh.meshtags(domain, tdim, all_cells_idx, markers)

    # === 5. 경계 조건 설정 (좌표 기반) ===
    if rank == 0:
        print("[Sim Func - Step 5] Setting up boundary conditions...")

    fdim = domain.topology.dim - 1

    # --- 좌표 기반 경계면 정의 헬퍼 함수 (단위: m) ---
    # 6개의 전체 벽
    def left_wall(x): return np.isclose(x[0], 0.0)  # X = -0.31
    def right_wall(x): return np.isclose(x[0], 0.31)  # X = 0
    def bottom_wall(x): return np.isclose(x[1], 0.0)  # Y = 0
    def top_wall(x): return np.isclose(x[1], 0.26)  # Y = 0.26
    def back_wall(x): return np.isclose(x[2], 0.0)  # Z = -0.31
    def front_wall(x): return np.isclose(x[2], 0.31)  # Z = 0

    portLS_y_min, portLS_y_max = 0.1075, 0.1525  # port Left short
    portBS_y_min, portBS_y_max = 0.1075, 0.1525

    # [수정] Z축(짧은 변): 15cm 폭 (중앙 정렬: 0.08 ~ 0.23)
    portTS_z_min, portTS_z_max = 0.08, 0.23

    portLL_z_min, portLL_z_max = 0.11, 0.2
    portBL_x_min, portBL_x_max = 0.11, 0.2

    # [수정] X축(긴 변): 31cm 전체 폭 (0.0 ~ 0.31)
    portTL_x_min, portTL_x_max = 0.0, 0.31


    # --- 경계면(Facet) 찾기 ---

    # a) 도체 경계 조건 (PEC)
    pec_facets_R = mesh.locate_entities_boundary(domain, fdim, right_wall)
    pec_facets_Ba = mesh.locate_entities_boundary(domain, fdim, back_wall)
    pec_facets_F = mesh.locate_entities_boundary(domain, fdim, front_wall)
    pec_facets_L = mesh.locate_entities_boundary(domain, fdim, left_wall)
    pec_facets_Bo = mesh.locate_entities_boundary(domain, fdim, bottom_wall)
    pec_facets_T = mesh.locate_entities_boundary(domain, fdim, top_wall)
    pec_facets = np.concatenate([pec_facets_R, pec_facets_Ba, pec_facets_F,pec_facets_L,pec_facets_Bo,pec_facets_T])

    pec_dofs = fem.locate_dofs_topological(V, fdim, pec_facets)
    u_pec = fem.Function(V)  # 기본값 0
    bc_pec = fem.dirichletbc(u_pec, pec_dofs)

    # b) 포트(Port) 경계 조건
    def portt_region(x):
        on_wall = np.isclose(x[1], 0.26)
        in_z = np.logical_and(x[2] >= portTS_z_min, x[2] <= portTS_z_max)
        in_x = np.logical_and(x[0] >= portTL_x_min, x[0] <= portTL_x_max)
        return np.logical_and(on_wall, np.logical_and(in_x, in_z))

    portt_facets = mesh.locate_entities_boundary(domain, fdim, portt_region)

    # --- 포트 파라미터 정의 (m 단위) ---
    E0 = 20000.0

    def port_field_func_T(x):
        values = np.zeros((3, x.shape[1]), dtype=np.complex128)
        width_long = portTL_x_max - portTL_x_min
        values[PortTA, :] = E0 * np.sin(np.pi * (x[PortTB] - portTL_x_min) / width_long) #여기서 - 0,2 2,0 은 수직방향
        return values

    # --- Weak Form Setup ---
    # Create MeshTags for boundaries to use in ds
    # PEC ID = 3, Port ID = 6
    ft_indices = np.concatenate([pec_facets, portt_facets])
    ft_values = np.concatenate([np.full(len(pec_facets), 3, dtype=np.int32),
                                np.full(len(portt_facets), 6, dtype=np.int32)])
    sorted_args = np.argsort(ft_indices)
    facet_tags = mesh.meshtags(domain, fdim, ft_indices[sorted_args], ft_values[sorted_args])
    
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    n = ufl.FacetNormal(domain)

    # Incident Field for Weak Form
    E_inc = fem.Function(V)
    E_inc.interpolate(port_field_func_T)

    # Only PEC BC in the list
    bcs = [bc_pec]

    if rank == 0:
        print("[Sim Func - Step 5] Finished setting up boundary conditions (Weak Form Port).")

    # === 6. Variational Formulation (약한 형식) ===
    if rank == 0:
        print("[Sim Func - Step 6] Defining the variational problem (weak form)...")

    E, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    a = ufl.inner(1 / mu_r * ufl.curl(E), ufl.curl(v)) * ufl.dx \
        - k0 ** 2 * ufl.inner(eps_r * E, v) * ufl.dx

    L = ufl.inner(fem.Constant(domain, np.array([0, 0, 0], dtype=np.complex128)), v) * ufl.dx

    # Add Weak Form Terms for Port (ID 6)
    # LHS: Impedance Boundary Condition
    a += 1j * k0 * ufl.inner(ufl.cross(n, E), ufl.cross(n, v)) * ds(6)
    # RHS: Excitation
    L += 2 * 1j * k0 * ufl.inner(ufl.cross(n, E_inc), ufl.cross(n, v)) * ds(6)

    if rank == 0:
        print("[Sim Func - Step 6] Finished defining variational problem.")

    # === 7. 선형 시스템 풀이 ===
    if rank == 0:
        print("[Sim Func - Step 7] Setting up and solving the linear problem...")

    E_h = fem.Function(V)
    E_h.name = "E_field"

    try:
        problem = fem.petsc.LinearProblem(a, L, bcs=bcs, u=E_h, petsc_options=petsc_options_em,petsc_options_prefix=petsc_options_prefix_em)
        problem.solve()
        if rank == 0:
            print("[Sim Func - Step 7] Finished solving.")
            print("[Debug] Solver converged. Iterations:", problem.solver.getIterationNumber())
    except Exception as e:
        if rank == 0:
            print("[Error] Solver failed:", str(e))
        return None, None, None

    # === 8. 후처리: 열원(Heat Source) 계산 ===
    # ... (Step 7: Solver 부분까지는 기존 코드 유지) ...

    # === 8. 후처리: 열원(Heat Source), 자기장, 포인팅 벡터, S11 계산 ===
    if rank == 0:
        print("[Sim Func - Step 8] Post-processing: Calculating Q, H, S, and Power-based S11...")

    # 1. Heat Source (Q) 계산
    V_Q = fem.functionspace(domain, ("DG", 0))
    Q = fem.Function(V_Q)
    Q.name = "HeatSource"
    # Q = 0.5 * omega * eps0 * imag(eps_r) * |E|^2
    Q_expr_ufl = 0.5 * omega * eps0 * (-ufl.imag(eps_r)) * ufl.real(ufl.dot(E_h, ufl.conj(E_h)))
    Q_expr = fem.Expression(Q_expr_ufl, V_Q.element.interpolation_points)
    Q.interpolate(Q_expr)

    # 2. E-field (Lagrange 공간으로 보간 - 시각화용)
    V_lagrange = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    E_h_lagrange = fem.Function(V_lagrange)
    E_h_lagrange.name = "E_field_Lagrange"
    E_h_expr = fem.Expression(E_h, V_lagrange.element.interpolation_points)
    E_h_lagrange.interpolate(E_h_expr)

    # 3. H-field (자기장) 계산: H = (-1 / j*w*mu) * curl(E)
    V_H = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    H_h = fem.Function(V_H)
    H_h.name = "H_field"
    H_ufl = (-1.0 / (1j * omega * mu0)) * ufl.curl(E_h)
    H_expr = fem.Expression(H_ufl, V_H.element.interpolation_points)
    H_h.interpolate(H_expr)

    # 4. Poynting Vector (에너지 흐름) 계산: S = 0.5 * Re(E x H*)
    V_S = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    S_vec = fem.Function(V_S)
    S_vec.name = "Poynting_Vector"
    S_ufl = 0.5 * ufl.real(ufl.cross(E_h, ufl.conj(H_h)))
    S_expr = fem.Expression(S_ufl, V_S.element.interpolation_points)
    S_vec.interpolate(S_expr)

    # -------------------------------------------------------------------------
    # [중요] 전력 기반 S11 계산 및 700W 스케일링
    # -------------------------------------------------------------------------

    # A. Port 입출력 전력 계산 (Power Flux)
    # Port ID는 6번으로 설정되어 있음 (ds(6))
    flux_term = ufl.dot(S_ufl, n)  # n은 도메인 바깥을 향함

    P_in_net = 0.0
    S11_dB = 0.0
    S11_complex = 0.0 + 0.0j

    try:
        # Port를 통해 나가는 에너지 적분 (보통 들어오는 에너지는 음수로 잡힘)
        P_flux_form = fem.form(flux_term * ds(6))
        P_net_out = comm.allreduce(fem.assemble_scalar(P_flux_form), op=MPI.SUM)

        # 우리가 관심 있는 건 "들어가는" 순수 전력
        P_in_net = -P_net_out

        # 입사 전력(Incident Power) 추정 (Analytical)
        # E_inc는 Step 5에서 정의됨. |E_inc|^2 / (2*Z0) 적분
        Z0 = 376.73  # 자유공간 임피던스
        E_inc_sq = ufl.inner(E_inc, E_inc)  # E_inc는 실수라고 가정 (위상 0)
        P_inc_form = fem.form((1.0 / (2.0 * Z0)) * E_inc_sq * ds(6))
        P_inc = comm.allreduce(fem.assemble_scalar(P_inc_form), op=MPI.SUM)

        # 반사 전력: P_ref = P_inc - P_in_net
        P_ref = P_inc - P_in_net

        # S11 (Magnitude) = sqrt(P_ref / P_inc)
        if P_inc > 1e-12:
            ratio = P_ref / P_inc
            if ratio < 0: ratio = 0
            s11_mag = np.sqrt(ratio)
            S11_dB = 20 * np.log10(s11_mag + 1e-16)

            # 위상은 E-field 평균에서 가져옴 (약식)
            # E_total / E_inc - 1 의 위상 사용
            e_tot_form = fem.form(E_h[1] * ds(6))  # Y성분 기준 (PortTA=1 가정)
            e_inc_form = fem.form(E_inc[1] * ds(6))
            val_tot = comm.allreduce(fem.assemble_scalar(e_tot_form), op=MPI.SUM)
            val_inc = comm.allreduce(fem.assemble_scalar(e_inc_form), op=MPI.SUM)
            if abs(val_inc) > 1e-12:
                complex_ratio = (val_tot / val_inc) - 1.0
                S11_complex = s11_mag * np.exp(1j * np.angle(complex_ratio))
            else:
                S11_complex = s11_mag + 0j

    except Exception as e:
        if rank == 0: print(f"[Warning] S11 Power calc failed: {e}")

    # B. 700W 스케일링 (음식물에 흡수된 전력 기준)
    target_power = 700.0
    total_absorbed_power = 0.0

    # pork_id = 1 (Step 2에서 정의됨)
    dx_sub = ufl.Measure("dx", domain=domain, subdomain_data=subdomains)

    try:
        # Q를 pork 영역(1)에 대해 적분
        power_form = fem.form(Q * dx_sub(1))
        total_absorbed_power = comm.allreduce(fem.assemble_scalar(power_form), op=MPI.SUM)
    except Exception as e:
        if rank == 0: print(f"[Warning] Power integration failed: {e}")

    scale_factor = 1.0
    if total_absorbed_power > 1e-12:
        scale_factor = target_power / total_absorbed_power
        if rank == 0: print(
            f"[Sim Func] Scaling results: Sim Power {total_absorbed_power:.4f} W -> Target {target_power} W (Factor: {scale_factor:.4e})")

        # 필드 스케일링 적용
        Q.x.array[:] *= scale_factor
        S_vec.x.array[:] *= scale_factor

        sqrt_scale = np.sqrt(scale_factor)
        E_h_lagrange.x.array[:] *= sqrt_scale
        H_h.x.array[:] *= sqrt_scale

        # P_in_net도 스케일링
        P_in_net *= scale_factor
    else:
        if rank == 0: print("[Warning] Total absorbed power is 0 or negative. Skipping scaling.")

    # -------------------------------------------------------------------------
    # CSV 저장
    # -------------------------------------------------------------------------
    # Port 평균 벡터 계산 (저장용)
    E_vec_avg = np.zeros(3, dtype=np.complex128)
    try:
        area_form = fem.form(fem.Constant(domain, petsc4py.PETSc.ScalarType(1.0)) * ds(6))
        port_area = comm.allreduce(fem.assemble_scalar(area_form), op=MPI.SUM)
        if port_area > 1e-9:
            for i in range(3):
                val = comm.allreduce(fem.assemble_scalar(fem.form(E_h[i] * ds(6))), op=MPI.SUM)
                E_vec_avg[i] = val / port_area
                # 스케일링 반영
                E_vec_avg[i] *= (np.sqrt(scale_factor) if total_absorbed_power > 1e-12 else 1.0)
    except:
        pass

    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        csv_path = os.path.join(output_dir, "s_parameters.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a") as f:
            if not file_exists:
                f.write("Sim_Number,Freq_Hz,S11_Real,S11_Imag,S11_Mag_dB,S11_Phase_Rad,"
                        "Ex_Real,Ex_Imag,Ey_Real,Ey_Imag,Ez_Real,Ez_Imag,Net_Power_Watt\n")

            s11_phase = np.angle(S11_complex)
            f.write(f"{number},{freq},{S11_complex.real:.6e},{S11_complex.imag:.6e},{S11_dB:.6f},{s11_phase:.6f},"
                    f"{E_vec_avg[0].real:.6e},{E_vec_avg[0].imag:.6e},"
                    f"{E_vec_avg[1].real:.6e},{E_vec_avg[1].imag:.6e},"
                    f"{E_vec_avg[2].real:.6e},{E_vec_avg[2].imag:.6e},"
                    f"{P_in_net:.6e}\n")

        print(f"[Debug] S11: {S11_dB:.2f} dB")
        print(f"[Debug] Net Power into Port (Scaled): {P_in_net:.4f} W")

    if rank == 0:
        print("[Sim Func - Step 8] Finished post-processing.")

    # === 9. 결과 저장 (XDMF) ===
    if rank == 0:
        print(f"[Sim Func - Step 9] Saving results to {output_dir}/ ...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    comm.Barrier()

    output_filename = os.path.join(output_dir, f"{number}.xdmf")
    with io.XDMFFile(comm, output_filename, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(subdomains, domain.geometry)

        # E, H, Q, 그리고 Poynting Vector(S) 저장
        xdmf.write_function(E_h_lagrange)
        xdmf.write_function(H_h)
        xdmf.write_function(S_vec)  # [중요] 에너지 흐름 벡터
        xdmf.write_function(Q)

    if rank == 0:
        print(f"[Sim Func - Step 9] Finished saving results to {output_filename}.")

    return E_h_lagrange, Q, E_vec_avg

import itertools
import pprint
# --- 이 스크립트가 메인으로 실행될 때 (테스트용) ---
if __name__ == "__main__":

    # 이 블록은 'mpirun -n [N] python grok2_callable.py'로 직접 실행할 때만 동작합니다.
    comm = MPI.COMM_WORLD
    rank = comm.rank

    PortLA = 1
    PortLB = 2
    PortBA = 1
    PortBB = 0
    PortTB = 0 #
    PortTA = 2 #
    
    petsc_options_prefix_em = "pc"
    current_petsc_options = {
        # 1. 반복법을 사용하지 않고 프리컨디셔너(LU)만 한 번 적용
        "ksp_type": "preonly",
        "pc_type": "lu",

        # 2. LU 분해를 수행할 외부 솔버로 mumps 지정
        "pc_factor_mat_solver_type": "mumps",

        # 3. MUMPS 전용 옵션 (메모리 부족 시 하드디스크 활용 설정 포함)
        "mat_mumps_icntl_22": 1,
        "mat_mumps_ooc_tmpdir": os.getcwd(),
        "mat_mumps_icntl_14": 200,  # 행렬 분해 시 메모리 여유분 (기본값보다 높게 설정)

        # 4. 모니터링 (직접법이므로 잔차가 나오지 않고 바로 완료됨)
        "ksp_view": None,
        "ksp_converged_reason": None,
    }

    now = time.time()
    E_field, Q_source, E_vec = run_microwave_simulation(
        number="lu_test2_deg2_hex",
        n_elem_x=int(31*1),
        n_elem_y=int(26*1),
        n_elem_z=int(31*1),
        output_dir="mesh_find",
        comm=comm,
        PortLA=PortLA, PortLB=PortLB,
        PortTA=PortTA, PortTB=PortTB,
        PortBA=PortBA, PortBB=PortBB,
        degree=2, # 2필수
        petsc_options_em=current_petsc_options,
        petsc_options_prefix_em=petsc_options_prefix_em,
        cell_hex_tet ="tet"
    )
    now = time.time() - now
    hours = int(now // 3600)
    minutes = int((now % 3600) // 60)
    seconds = int(now % 60)

    if rank == 0:
        print(f"걸린 시간: {hours}시간 {minutes}분 {seconds}초")
        print(f"Port Vector: {E_vec}")
        print_peak_memory()
