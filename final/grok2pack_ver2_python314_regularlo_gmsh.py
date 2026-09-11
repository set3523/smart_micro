"""
DOLFINx-based EM solver for Microwave Oven Simulation (Gmsh Integration Version)
이 스크립트는 Gmsh API를 사용하여 형상을 생성하고 메쉬를 로드합니다.
"""

import petsc4py.PETSc
from dolfinx import fem, mesh, io
import dolfinx.fem.petsc
from dolfinx.io import gmsh  as dolfinx_gmsh# 수정: gmsh 모듈을 명시적으로 임포트

# 병렬 계산을 위한 MPI
from mpi4py import MPI

# 방정식을 정의하기 위한 UFL
import ufl

# NumPy
import basix.ufl
import numpy as np

# Gmsh
import gmsh

# 경로 관리를 위한 os
import os
import time
import resource
import pprint


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


def generate_microwave_mesh(comm, model_rank, n_elem_x, n_elem_y, n_elem_z):
    """
    Gmsh API를 사용하여 형상을 만들고 메쉬를 생성합니다.
    오차 보정(mesh_tol)을 제거하여 정사이즈로 생성합니다.
    """
    gmsh.initialize()
    
    if comm.rank == model_rank:
        # gmsh.model.add("microwave_pork_gmsh")

        # === 1. 파라미터 설정 ===
        chamber_size = [0.31, 0.26, 0.31]

        # 메쉬 해상도 기반 dx, dy, dz 계산 (메쉬 사이즈 결정용)
        dx = chamber_size[0] / n_elem_x
        dy = chamber_size[1] / n_elem_y
        dz = chamber_size[2] / n_elem_z

        # Gmsh용 특성 길이 (Characteristic Length)
        lc_min = min(dx, dy, dz)

        # [수정] 오차 보정값 제거 (0.0으로 설정)
        mesh_tol = 0.0

        # 치수 정의 (보정 없이 원래 크기 사용)
        base_radius = 0.04
        eff_radius = base_radius + mesh_tol

        cube_side = 0.09
        cube_half = cube_side / 2.0
        eff_cube_half = cube_half + mesh_tol
        eff_cube_full = eff_cube_half * 2.0

        y_center_bottom = 0.08
        y_center_top = 0.18
        y_center_cube = 0.19

        # === 2. 형상 생성 (OpenCASCADE) ===

        # (1) 챔버 (Outer Box)
        chamber_tag = gmsh.model.occ.addBox(0, 0, 0, chamber_size[0], chamber_size[1], chamber_size[2])

        # port_x = 0.11
        # port_y = 0.26
        # port_z = 0.1325
        # port_dx = 0.2 - 0.11
        # port_dz = 0.1775 - 0.1325

        port_x = 0.0
        port_y = 0.26
        port_z = 0.08
        port_dx = 0.31  # 31cm (물리적 한계로 인해 줄일 수 없음)
        port_dz = 0.15  # 15cm (표준 비율 2:1 적용)

        p1 = gmsh.model.occ.addPoint(port_x, port_y, port_z)
        p2 = gmsh.model.occ.addPoint(port_x + port_dx, port_y, port_z)
        p3 = gmsh.model.occ.addPoint(port_x + port_dx, port_y, port_z + port_dz)
        p4 = gmsh.model.occ.addPoint(port_x, port_y, port_z + port_dz)

        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p4)
        l4 = gmsh.model.occ.addLine(p4, p1)

        port_loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        port_surface_tag = gmsh.model.occ.addPlaneSurface([port_loop])

        pork_tags = []

        # (2) 구 (Spheres)
        sphere_coords = [
            (0.105, y_center_bottom, 0.105),
            (0.205, y_center_bottom, 0.205),
            (0.105, y_center_top, 0.119),
            (0.105, y_center_top, 0.200)
        ]
        for cx, cy, cz in sphere_coords:
            tag = gmsh.model.occ.addSphere(cx, cy, cz, eff_radius)
            pork_tags.append(tag)

        # (3) 정육면체 (Cubes)
        cube_centers = [(0.205, 0.105), (0.205, 0.205)]
        for cx, cz in cube_centers:
            x0 = cx - eff_cube_half
            y0 = y_center_cube - eff_cube_half
            z0 = cz - eff_cube_half
            tag = gmsh.model.occ.addBox(x0, y0, z0, eff_cube_full, eff_cube_full, eff_cube_full)
            pork_tags.append(tag)

        # (4) 정사면체 (Tetrahedrons)
        H = 2.0 * eff_radius
        r_base = H / np.sqrt(2)
        v_top = np.array([0.0, eff_radius, 0.0])
        v_b1 = np.array([r_base, -eff_radius, 0.0])
        v_b2 = np.array([r_base * np.cos(2 * np.pi / 3), -eff_radius, r_base * np.sin(2 * np.pi / 3)])
        v_b3 = np.array([r_base * np.cos(4 * np.pi / 3), -eff_radius, r_base * np.sin(4 * np.pi / 3)])
        verts_up = [v_top, v_b1, v_b2, v_b3]
        verts_down = [v * np.array([1.0, -1.0, 1.0]) for v in verts_up]

        def add_tetrahedron_occ(vertices, offset):
            p_tags = []
            for v in vertices:
                x, y, z = v[0] + offset[0], v[1] + offset[1], v[2] + offset[2]
                p_tags.append(gmsh.model.occ.addPoint(x, y, z))

            face_indices = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
            face_tags = []

            for idxs in face_indices:
                # 각 면마다 선과 루프를 생성
                l1 = gmsh.model.occ.addLine(p_tags[idxs[0]], p_tags[idxs[1]])
                l2 = gmsh.model.occ.addLine(p_tags[idxs[1]], p_tags[idxs[2]])
                l3 = gmsh.model.occ.addLine(p_tags[idxs[2]], p_tags[idxs[0]])
                wire = gmsh.model.occ.addCurveLoop([l1, l2, l3])
                face = gmsh.model.occ.addPlaneSurface([wire])
                face_tags.append(face)

                # [중요] 생성된 면들을 fragment로 처리하여 모서리를 공유하게 만듦
                # 이렇게 해야 위상적으로 연결되어 닫힌 쉘(Closed Shell)을 만들 수 있음
            f_dimtags = [(2, t) for t in face_tags]
            out, _ = gmsh.model.occ.fragment(f_dimtags, [])

            # fragment 결과에서 면 태그들만 추출
            new_f_tags = [t[1] for t in out if t[0] == 2]

            # 쉘(Shell) 생성 후 부피(Volume) 생성
            shell = gmsh.model.occ.addSurfaceLoop(new_f_tags)
            solid = gmsh.model.occ.addVolume([shell])

            return solid

        t_up = add_tetrahedron_occ(verts_up, (0.105, y_center_bottom, 0.205))
        pork_tags.append(t_up)
        t_down = add_tetrahedron_occ(verts_down, (0.205, y_center_bottom, 0.105))
        pork_tags.append(t_down)
        gmsh.model.occ.synchronize()
        # gmsh.write("microwave_geometry_debug3.step")

        # === 3. Boolean 연산 (Fragment 사용) ===
        chamber_dimtag = [(3, chamber_tag)]
        pork_dimtags = [(3, t) for t in pork_tags]
        tool_dimtags = pork_dimtags + [(2, port_surface_tag)]

        # fragment 연산 수행 (Conformal Mesh 보장)

        gmsh.model.occ.fragment(chamber_dimtag, tool_dimtags)
        #gmsh.model.occ.fragment(chamber_dimtag, pork_dimtags)
        gmsh.model.occ.synchronize()
        gmsh.model.geo.synchronize() # [추가] Geo 모델 동기화
        # gmsh.write("microwave_geometry_debug2.step")

        mass_id = 1
        air_id = 2
        port_id = 3
        wall_id = 4

        # === 4. Physical Group 설정 ===
        all_volumes = gmsh.model.getEntities(3)
        air_volumes = []
        pork_volumes = []

        for dim, tag in all_volumes:
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            # mass 영역 판별 로직 (위치 및 크기 기반)
            if abs(com[0] - 0.155) < 0.1 and abs(com[1] - 0.14) < 0.1:
                bbox = gmsh.model.getBoundingBox(dim, tag)
                vol_dx = bbox[3] - bbox[0]
                if vol_dx < 0.30:  # 챔버 전체 크기보다 작으면 Pork
                    pork_volumes.append(tag)
                else:
                    air_volumes.append(tag)
            else:
                air_volumes.append(tag)

        pork_id, air_id = 1, 2
        if pork_volumes:
            gmsh.model.addPhysicalGroup(3, pork_volumes, pork_id, name="Pork")
        if air_volumes:
            gmsh.model.addPhysicalGroup(3, air_volumes, air_id, name="Air")

        all_surfaces = gmsh.model.getEntities(2)
        port_surfaces = []
        wall_surfaces = []

        # Port 영역 판별 기준
        tol = 1e-4
        target_y = 0.26
        target_x_min, target_x_max = 0.11, 0.2
        target_z_min, target_z_max = 0.1325, 0.1775

        # 챔버 경계 기준 (Wall 판별용)
        chamber_min = [0.0, 0.0, 0.0]
        chamber_max = chamber_size  # [0.31, 0.26, 0.31]

        for dim, tag in all_surfaces:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            # bbox: [minX, minY, minZ, maxX, maxY, maxZ]

            cx = (bbox[0] + bbox[3]) / 2.0
            cy = (bbox[1] + bbox[4]) / 2.0
            cz = (bbox[2] + bbox[5]) / 2.0

            is_port = False

            # 1. Port 확인 (Top Wall에 있고, X/Z 범위 내)
            if abs(bbox[1] - target_y) < tol and abs(bbox[4] - target_y) < tol:
                if (target_x_min - tol <= cx <= target_x_max + tol) and \
                        (target_z_min - tol <= cz <= target_z_max + tol):
                    # 너무 작은 조각(노이즈) 제외
                    if (bbox[3] - bbox[0]) > 0.01 and (bbox[5] - bbox[2]) > 0.01:
                        port_surfaces.append(tag)
                        is_port = True

            # 2. Wall 확인 (Port가 아니면서, 챔버 외곽 경계에 있는 면)
            if not is_port:
                # 6면 중 하나라도 닿아 있으면 Wall로 간주
                on_boundary = False
                # X축 경계 (Left/Right)
                if abs(bbox[0] - chamber_min[0]) < tol or abs(bbox[3] - chamber_max[0]) < tol:
                    on_boundary = True
                # Y축 경계 (Bottom/Top)
                elif abs(bbox[1] - chamber_min[1]) < tol or abs(bbox[4] - chamber_max[1]) < tol:
                    on_boundary = True
                # Z축 경계 (Back/Front)
                elif abs(bbox[2] - chamber_min[2]) < tol or abs(bbox[5] - chamber_max[2]) < tol:
                    on_boundary = True

                if on_boundary:
                    wall_surfaces.append(tag)

        if port_surfaces:
            gmsh.model.addPhysicalGroup(2, port_surfaces, port_id, name="port")
        if wall_surfaces:
            gmsh.model.addPhysicalGroup(2, wall_surfaces, wall_id, name="wall")

        # === 5. 메쉬 설정 ===
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_min)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        gmsh.model.mesh.clear() # [추가] 기존 메쉬 클리어
        gmsh.model.mesh.generate(3)
        
        # [추가] 메쉬 생성 검증
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(3)
        if not element_types or len(element_tags[0]) == 0:
            print("[ERROR] Mesh generation failed - no 3D elements created!")
        else:
            print(f"[Mesh] Generated {len(node_tags[0])} elements in 3D")

    # DOLFINx로 메쉬 로드
    # [수정] dolfinx.io.gmsh 모듈(별칭 dolfinx_gmsh)을 사용하여 메쉬 로드
    model_output = dolfinx_gmsh.model_to_mesh(
        gmsh.model, comm, rank=model_rank, gdim=3
    )
    domain = model_output[0]
    cell_tags = model_output[1]
    facet_tags = model_output[2]

    gmsh.finalize()

    return domain, cell_tags, facet_tags


def run_microwave_simulation(
        number="checkcheck",
        n_elem_x=93,
        n_elem_y=84,
        n_elem_z=93,
        output_dir="result",
        comm=MPI.COMM_WORLD,
        PortLA=1, PortLB=1,
        PortTA=1, PortTB=1,
        PortBA=1, PortBB=1,
        degree=1,
        petsc_options_em={},
        petsc_options_prefix_em="test",
):
    # === 0. MPI 커뮤니케이터 설정 ===
    rank = comm.rank
    if rank == 0:
        pprint.pprint(petsc_options_em)

    # === 1. 설정 및 메쉬 생성 (Gmsh 사용) ===
    if rank == 0:
        print(f"[Sim Func - Step 1] Generating mesh with Gmsh... (Target Res: {n_elem_x}x{n_elem_y}x{n_elem_z})")

    domain, cell_tags, facet_tags = generate_microwave_mesh(comm, 0, n_elem_x, n_elem_y, n_elem_z)

    # Gmsh에서 정의한 ID
    mass_id = 1
    air_id = 2
    port_id = 3
    wall_id = 4

    # === 2. 물리 상수 및 재료 물성 정의 ===
    if rank == 0:
        print("[Sim Func - Step 2] Defining physical constants and material properties...")

    freq = 2.45e9 # 일단 ㅜ언래값
    freq = 2.45e9/5  # 일단 ㅜ언래값
    omega = 2 * np.pi * freq
    c0 = 299792458.0
    mu0 = 4 * np.pi * 1e-7
    eps0 = 1 / (mu0 * c0 ** 2)
    k0 = omega / c0

    eps_r_mass_re = 49.0
    tand_mass = 0.3
    eps_r_mass = eps_r_mass_re * (1 - 1j * tand_mass)

    # Air 물성
    eps_r_air = 1.0 + 0.0j
    mu_r_mass = 1.0 + 0.0j
    mu_r_air = 1.0 + 0.0

    # === 3. 함수 공간 정의 ===
    if rank == 0:
        print("[Sim Func - Step 3] Defining function spaces...")

    V = fem.functionspace(domain, ("N1curl", degree))
    V_mat = fem.functionspace(domain, ("DG", 0))

    # === 4. 재료 물성 함수 설정 (Gmsh 태그 기반) ===
    if rank == 0:
        print("[Sim Func - Step 4] Setting up material property functions using Gmsh tags...")

    eps_r = fem.Function(V_mat)
    eps_r.name = "epsilon_r"
    mu_r = fem.Function(V_mat)
    mu_r.name = "mu_r"

    eps_r.x.array[:] = eps_r_air
    mu_r.x.array[:] = mu_r_air

    mass_cells = cell_tags.indices[cell_tags.values == mass_id]
    if len(mass_cells) > 0:
        eps_r.x.array[mass_cells] = eps_r_mass
        mu_r.x.array[mass_cells] = mu_r_mass

    subdomains = cell_tags

    # === 5. 경계 조건 설정 ===
    if rank == 0:
        print("[Sim Func - Step 5] Setting up boundary conditions (Weak Form Port)...")

    fdim = domain.topology.dim - 1

    wall_facets = facet_tags.indices[facet_tags.values == wall_id]

    if len(wall_facets) > 0:
        wall_dofs = fem.locate_dofs_topological(V, fdim, wall_facets)
        u_pec = fem.Function(V)
        u_pec.x.array[:] = 0.0
        bc_pec = fem.dirichletbc(u_pec, wall_dofs)
    else:
        if rank == 0: print("[Warning] No wall facets found!")
        bc_pec = None

    # (2) Port 경계 조건 (Weak Form)
    # Gmsh에서 'port'로 태깅된 면들을 가져옴
    port_facets = facet_tags.indices[facet_tags.values == port_id]

    # portTL_x_min, portTL_x_max = 0.11, 0.2
    portTL_x_min, portTL_x_max = 0.0, 0.31

    E0 = 20000.0
    # Port 필드 함수 (좌표 기반 계산 유지)
    def port_field_func_T(x):
        values = np.zeros((3, x.shape[1]), dtype=np.complex128)
        width_long = portTL_x_max - portTL_x_min
        # x[PortTB]는 좌표축 인덱스 (PortTB=0 이면 x축)
        values[PortTA, :] = E0 * np.sin(np.pi * (x[PortTB] - portTL_x_min) / width_long)
        return values

    # Incident Field for Weak Form
    E_inc = fem.Function(V)
    E_inc.interpolate(port_field_func_T)

    bcs = []
    if bc_pec: bcs.append(bc_pec)
    # Port Dirichlet BC removed

    # === 6. Variational Formulation ===
    E, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    n = ufl.FacetNormal(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    a = ufl.inner(1 / mu_r * ufl.curl(E), ufl.curl(v)) * ufl.dx \
        - k0 ** 2 * ufl.inner(eps_r * E, v) * ufl.dx

    L = ufl.inner(fem.Constant(domain, np.array([0, 0, 0], dtype=np.complex128)), v) * ufl.dx

    # Add Weak Form Terms for Port (ID 3)
    if len(port_facets) > 0:
        # LHS: Impedance Boundary Condition
        a += 1j * k0 * ufl.inner(ufl.cross(n, E), ufl.cross(n, v)) * ds(port_id)
        # RHS: Excitation
        L += 2 * 1j * k0 * ufl.inner(ufl.cross(n, E_inc), ufl.cross(n, v)) * ds(port_id)

    # === 7. Solver ===
    if rank == 0:
        print("[Sim Func - Step 7] Solving...")

    E_h = fem.Function(V)
    E_h.name = "E_field"

    try:
        problem = fem.petsc.LinearProblem(a, L, bcs=bcs, u=E_h, petsc_options=petsc_options_em,
                                          petsc_options_prefix=petsc_options_prefix_em)
        problem.solve()
    except Exception as e:
        if rank == 0: print("[Error] Solver failed:", str(e))
        return None, None, None

    # === 8. Post-processing ===
    if rank == 0: print("[Sim Func - Step 8] Calculating Q...")

    V_Q = fem.functionspace(domain, ("DG", 0))
    Q = fem.Function(V_Q)
    Q.name = "HeatSource"
    Q_expr_ufl = 0.5 * omega * eps0 * (-ufl.imag(eps_r)) * ufl.real(ufl.dot(E_h, ufl.conj(E_h)))
    Q_expr = fem.Expression(Q_expr_ufl, V_Q.element.interpolation_points)
    Q.interpolate(Q_expr)

    V_lagrange = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    E_h_lagrange = fem.Function(V_lagrange)
    E_h_lagrange.name = "E_field_Lagrange"
    E_h_expr = fem.Expression(E_h, V_lagrange.element.interpolation_points)
    E_h_lagrange.interpolate(E_h_expr)

    # Calculate Port Vector Average
    E_vec_avg = np.zeros(3, dtype=np.complex128)
    try:
        if len(port_facets) > 0:
            area_form = fem.form(fem.Constant(domain, petsc4py.PETSc.ScalarType(1.0)) * ds(port_id))
            port_area = comm.allreduce(fem.assemble_scalar(area_form), op=MPI.SUM)
            if port_area > 1e-9:
                for i in range(3):
                    e_comp_form = fem.form(E_h[i] * ds(port_id))
                    val = comm.allreduce(fem.assemble_scalar(e_comp_form), op=MPI.SUM)
                    E_vec_avg[i] = val / port_area
    except Exception as e:
        if rank == 0: print(f"[Warning] Port vector calc failed: {e}")

    if rank == 0:
        print(f"[Debug] Port E-field Avg: {E_vec_avg}")

    # === 9. Save ===
    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    comm.Barrier()
    output_filename = os.path.join(output_dir, f"{number}.xdmf")
    with io.XDMFFile(comm, output_filename, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(subdomains, domain.geometry)  # Cell tags (mass/air)
        xdmf.write_meshtags(facet_tags, domain.geometry)  # Facet tags (wall/port)
        xdmf.write_function(E_h_lagrange)
        xdmf.write_function(Q)

    return E_h_lagrange, Q, E_vec_avg


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.rank

    PortLA, PortLB = 1, 2
    PortBA, PortBB = 1, 0
    PortTB, PortTA = 0, 2

    petsc_options_prefix_em = "pc"
    current_petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_22": 1,
        "mat_mumps_ooc_tmpdir": os.getcwd(),
        "mat_mumps_icntl_14": 200,
        "ksp_view": None,
    }

    now = time.time()
    E_field, Q_source, E_vec = run_microwave_simulation(
        number="gmsh_test_tet",
        n_elem_x=31*2,
        n_elem_y=26*2,
        n_elem_z=31*2,
        output_dir="mesh_gmsh_result",
        comm=comm,
        PortLA=PortLA, PortLB=PortLB,
        PortTA=PortTA, PortTB=PortTB,
        PortBA=PortBA, PortBB=PortBB,
        degree=2,
        petsc_options_em=current_petsc_options,
        petsc_options_prefix_em=petsc_options_prefix_em,
    )

    if rank == 0:
        elapsed = time.time() - now
        print(f"Total Time: {elapsed:.2f} sec")
        print(f"Port Vector: {E_vec}")
        print_peak_memory()
