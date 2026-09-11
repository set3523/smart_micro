import gmsh
import numpy as np
import ufl
from dolfinx import fem, mesh, io, plot, geometry
from dolfinx.io import gmsh as dolfinx_gmsh
from mpi4py import MPI
from petsc4py import PETSc
import sys
import os
import dolfinx.fem.petsc
import h5py
import random
import csv

# -----------------------------------------------------------------------------
# 1. Geometry & Mesh Generation (Modularized) None glass version
# -----------------------------------------------------------------------------
freq = 2.45e9  # 전자레인지 주파수 (Hz)
omega = 2 * np.pi * freq
c0 = 299792458.0  # 진공 중 빛의 속도 (m/s)
mu0 = 4 * np.pi * 1e-7  # 진공 중 투자율 (H/m)
eps0 = 1 / (mu0 * c0 ** 2)  # 진공 중 유전율 (F/m)
k0 = omega / c0  # 진공 중 파수 (rad/m)
ref_file = "cal_by_gmsh3/microwave_geometry_and_stackdumpling_0.xdmf"


def create_mesh_from_geo(comm, geo_filename, mesh_min=0.003, mesh_max=0.015):
    """
    .geo 파일을 읽어서 메쉬를 생성하고, Physical Group 이름과 태그 매핑을 반환합니다.
    [추가] 'center'라는 Physical Point가 있으면 좌표를 추출하여 'rotation_center.txt'에 저장합니다.
    """
    gmsh.initialize()

    # Rank 0에서만 Gmsh 실행
    if comm.rank == 0:
        gmsh.open(geo_filename)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_max)
        # 모델 동기화 (OCC 또는 Built-in 커널)
        try:
            gmsh.model.occ.synchronize()
        except:
            pass
        gmsh.model.geo.synchronize()

        # 기존 메쉬가 있다면 제거 (태그 수정 후 재생성을 위해)
        gmsh.model.mesh.clear()

        # --- [Fix] Untagged Volume 처리 ---
        # 1. 모든 3D 엔티티(볼륨) 가져오기
        all_volumes = gmsh.model.getEntities(3)  # [(3, tag), (3, tag), ...]
        all_vol_tags = set(tag for dim, tag in all_volumes)

        # 2. 이미 Physical Group에 할당된 볼륨 태그 찾기
        tagged_vol_tags = set()
        phys_groups = gmsh.model.getPhysicalGroups(3)
        for p_dim, p_tag in phys_groups:
            entities = gmsh.model.getEntitiesForPhysicalGroup(p_dim, p_tag)
            for tag in entities:
                tagged_vol_tags.add(tag)

        # 3. 태그되지 않은 볼륨 찾기
        untagged_vols = list(all_vol_tags - tagged_vol_tags)

        print(f"[Mesh] Total volumes: {len(all_vol_tags)}")
        print(f"[Mesh] Tagged volumes: {len(tagged_vol_tags)}")
        print(f"[Mesh] Untagged volumes: {len(untagged_vols)}")

        # 4. 태그되지 않은 볼륨이 있다면 "Background_Air" 그룹에 추가
        if untagged_vols:
            print(f"[Mesh] Assigning {len(untagged_vols)} untagged volumes to 'Background_Air'.")

            # 기존에 Background_Air 그룹이 있는지 확인
            bg_tag = -1
            for p_dim, p_tag in phys_groups:
                if gmsh.model.getPhysicalName(p_dim, p_tag) == "Background_Air":
                    bg_tag = p_tag
                    break

            if bg_tag == -1:
                # 새 그룹 생성 (기존 태그 중 최대값 + 1)
                max_tag = max([t for d, t in phys_groups], default=0)
                bg_tag = max_tag + 1
                gmsh.model.addPhysicalGroup(3, untagged_vols, bg_tag, "Background_Air")
            else:
                # 기존 그룹에 추가
                existing_entities = gmsh.model.getEntitiesForPhysicalGroup(3, bg_tag)
                # 중복 제거하여 합치기
                new_entities = list(set(list(existing_entities) + untagged_vols))

                # 기존 그룹 정의를 지우고 다시 생성 (Gmsh API 특성상 덮어쓰기 위해)
                gmsh.model.removePhysicalGroups([(3, bg_tag)])
                gmsh.model.addPhysicalGroup(3, new_entities, bg_tag, "Background_Air")

        # ---------------------------------------------------------------------
        # [추가] Rotation Center 좌표 추출 및 저장 로직
        # .geo 파일에서 Physical Point("center", 1931)로 정의한 점을 찾습니다.
        # ---------------------------------------------------------------------
        try:
            # "center"라는 이름의 Physical Group ID 찾기
            center_tag = -1
            phys_groups_0 = gmsh.model.getPhysicalGroups(0)  # dim=0 (Point)
            for p_dim, p_tag in phys_groups_0:
                if gmsh.model.getPhysicalName(p_dim, p_tag) == "center":
                    center_tag = p_tag
                    break

            # 만약 이름으로 못 찾으면 ID 1931로 시도
            if center_tag == -1:
                center_tag = 1931

            # 해당 태그를 가진 노드 좌표 가져오기
            # getNodesForPhysicalGroup returns (nodeTags, coord)
            node_tags, coords = gmsh.model.mesh.getNodesForPhysicalGroup(0, center_tag)

            # 좌표가 아직 생성되지 않았을 수 있으므로, 엔티티에서 직접 좌표를 가져오거나
            # 메쉬 생성 전에 엔티티의 BoundingBox를 확인할 수도 있음.
            # 하지만 Point 엔티티는 메쉬 생성 전에도 좌표가 확정되어 있음.
            if len(coords) == 0:
                # 메쉬 노드가 없다면 엔티티 자체의 좌표를 조회 시도
                entities = gmsh.model.getEntitiesForPhysicalGroup(0, center_tag)
                if entities:
                    # 첫 번째 점 엔티티의 좌표 가져오기
                    bbox = gmsh.model.getBoundingBox(0, entities[0])
                    # bbox: [minx, miny, minz, maxx, maxy, maxz] -> 점이니까 min=max
                    coords = [bbox[0], bbox[1], bbox[2]]

            if len(coords) >= 3:
                # coords는 [x, y, z, ...] 형태
                center_coords = coords[:3]
                print(f"[Mesh] Found defined rotation center in Geo: {center_coords}")

                # 파일로 저장 (다른 함수에서 읽을 수 있게)
                np.savetxt("rotation_center.txt", center_coords)
            else:
                print("[Mesh] Warning: 'center' physical point defined but no coordinates found.")

        except Exception as e:
            print(f"[Mesh] Note: Could not extract rotation center from Gmsh: {e}")
        # ---------------------------------------------------------------------

        # 3D 메쉬 생성
        print("[Mesh] Generating 3D mesh...")
        gmsh.model.mesh.setOrder(2)
        gmsh.model.mesh.generate(3)

        # Validate mesh before conversion
        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(3)
        if not element_types or len(element_tags[0]) == 0:
            print("[ERROR] Mesh generation failed - no 3D elements created!")
            gmsh.finalize()
            sys.exit(1)
        print(f"[Mesh] Generated {len(node_tags[0])} elements in 3D")

    # Gmsh 모델을 DOLFINx 메쉬로 변환
    print("[Mesh] Converting to DOLFINx mesh...")
    model_output = dolfinx_gmsh.model_to_mesh(gmsh.model, comm, rank=0, gdim=3)
    domain = model_output[0]
    cell_tags = model_output[1]
    facet_tags = model_output[2]

    # Physical Group 이름과 ID 매핑 추출 (Rank 0에서 수행 후 브로드캐스트)
    name_to_id = {}
    print("===================\n")
    if comm.rank == 0:
        # 2D (Surface) 및 3D (Volume) 그룹 조회
        for dim in [2, 3]:
            physical_groups = gmsh.model.getPhysicalGroups(dim)
            for p_dim, p_tag in physical_groups:
                name = gmsh.model.getPhysicalName(p_dim, p_tag)
                if name:
                    name_to_id[name] = p_tag
                    print(f"Mapped '{name}' to ID {p_tag} (dim={p_dim})")

    print("===================\n")
    name_to_id = comm.bcast(name_to_id, root=0)

    gmsh.finalize()
    return domain, cell_tags, facet_tags, name_to_id

def create_mesh_from_step(comm, step_file, mesh_size_min=0.5, mesh_size_max=1.5):
    """
    STEP 파일을 읽어 메쉬를 생성합니다. (기존 함수 유지)
    """
    gmsh.initialize()
    if comm.rank == 0:
        gmsh.model.occ.importShapes(step_file)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(3)

    model_output = dolfinx_gmsh.model_to_mesh(gmsh.model, comm, rank=0, gdim=3)
    domain = model_output[0]
    cell_tags = model_output[1]
    facet_tags = model_output[2]
    gmsh.finalize()
    return domain, cell_tags, facet_tags, {} # 빈 맵 반환

# -----------------------------------------------------------------------------
# 2. Electromagnetic Solver
# -----------------------------------------------------------------------------

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


def solve_transient_heat_transfer_rotating(
        comm,
        schedule,
        rotation_center,  # (x, y, z) 회전 중심 좌표 (유리판 중심)
        output_file="result3/temperature_evolution.xdmf",
        dt=1.0,
        total_time=10.0,
        initial_temp=25.0
):
    """
    회전하는 메쉬의 Q 데이터를 고정된 메쉬(t=0)로 매핑하여 열 해석을 수행합니다.
    """

    # 1. 기준 메쉬(Reference Mesh) 로드 - t=0 (0도) 파일 사용
    # 열 해석은 이 고정된 메쉬 위에서 계속 진행됩니다.
    ref_file = schedule[0.0]  # 0초일 때 파일
    print(f"[HeatSolver] Loading REFERENCE mesh from {ref_file}...")

    with io.XDMFFile(comm, ref_file, "r") as xdmf:
        domain_ref = xdmf.read_mesh(name="mesh")
        cell_tags_ref = xdmf.read_meshtags(domain_ref, name="PhysicalGroups")

    if rotation_center is None:
        # Glass ID 찾기 (하드코딩 대신 변수 사용 권장, 여기선 1924 사용)
        id_glass = 1924
        glass_cells = cell_tags_ref.find(id_glass)

        if len(glass_cells) > 0:
            # Glass 셀들을 구성하는 모든 꼭짓점(Vertex) 좌표 가져오기
            domain_ref.topology.create_connectivity(domain_ref.topology.dim, 0)
            # entities_to_geometry: 셀 인덱스를 버텍스 인덱스로 변환
            vertex_indices = mesh.entities_to_geometry(domain_ref, domain_ref.topology.dim, glass_cells, True)
            unique_vertices = np.unique(vertex_indices)

            # 좌표 추출
            coords = domain_ref.geometry.x[unique_vertices]

            # Bounding Box 계산 (Min, Max)
            min_pt = np.min(coords, axis=0)
            max_pt = np.max(coords, axis=0)

            # 중심 계산 (Geo 파일의 로직과 동일)
            rotation_center = (min_pt + max_pt) / 2.0

            # 3D 좌표만 추출 (x, y, z)
            rotation_center = rotation_center[:3]

            if comm.rank == 0:
                print(f"[HeatSolver] Auto-calculated Rotation Center (Glass): {rotation_center}")
        else:
            # Glass가 없으면 기본값 사용 (에러 방지)
            rotation_center = (0.155, 0.13, 0.155)
            if comm.rank == 0:
                print("[HeatSolver] Warning: Glass domain not found. Using default rotation center.")


    # 2. Function Space 정의 (Reference Mesh 위에서)
    V_ref = fem.functionspace(domain_ref, ("Lagrange", 1))  # 온도용
    Q_space_ref = fem.functionspace(domain_ref, ("DG", 0))  # 열원용

    # 3. 물성치 및 초기 조건 설정 (이전과 동일)
    # ... (물성치 설정 코드는 동일하므로 생략, 필요시 이전 코드 복사) ...
    # 간단하게 Air/Glass/Mass ID를 찾아서 rho, cp, k 설정했다고 가정
    # -----------------------------------------------------------
    D0 = fem.functionspace(domain_ref, ("DG", 0))
    rho = fem.Function(D0)
    cp = fem.Function(D0)
    k_therm = fem.Function(D0)

    # 임시 물성치 (실제 ID에 맞게 수정 필요)
    rho.x.array[:] = 1.2
    cp.x.array[:] = 1000.0
    k_therm.x.array[:] = 0.026

    # ID 찾기 (예시)
    id_glass = 1924
    id_mass = 1925
    id_air = 1930

    def assign_prop(tag, r, c, k):
        cells = cell_tags_ref.find(tag)
        if len(cells) > 0:
            rho.x.array[cells] = r; cp.x.array[cells] = c; k_therm.x.array[cells] = k

    assign_prop(id_glass, 2500, 840, 1.0)
    assign_prop(id_mass, 1000, 3500, 0.5)

    # -----------------------------------------------------------

    # 초기 온도
    T_n = fem.Function(V_ref)
    T_n.name = "Temperature"
    T_n.x.array[:] = initial_temp

    # 4. 열 방정식 정의
    T = ufl.TrialFunction(V_ref)
    v = ufl.TestFunction(V_ref)
    Q_external = fem.Function(Q_space_ref)  # 매 스텝 업데이트 될 열원

    term_time = (rho * cp / dt) * ufl.inner(T - T_n, v) * ufl.dx

    # 2) 확산 항: k * dot(grad(T), grad(v))
    # ufl.dot 대신 ufl.inner를 사용하는 것이 안전합니다.
    term_diff = k_therm * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx

    # 3) 열원 항: Q * v
    term_source = ufl.inner(Q_external, v) * ufl.dx

    # 전체 방정식 F = 0
    F = term_time + term_diff - term_source

    petsc_options_prefix_heat = "pc_heat"  # (접두어 충돌 방지를 위해 이름 살짝 변경 추천)
    problem = fem.petsc.LinearProblem(ufl.lhs(F), ufl.rhs(F), bcs=[],
                                      petsc_options={"ksp_type": "cg", "pc_type": "gamg"},
                                      petsc_options_prefix=petsc_options_prefix_heat)

    # ... (이하 코드 동일) ...


    # 5. 결과 저장용 파일
    xdmf_out = io.XDMFFile(comm, output_file, "w")
    xdmf_out.write_mesh(domain_ref)
    xdmf_out.write_function(T_n, 0.0)

    # 6. 시간 루프
    t = 0.0
    print(f"[HeatSolver] Starting simulation...")

    # Q_space_ref의 보간점(Interpolation Points) 좌표 미리 가져오기
    # 이 점들이 회전하면서 값을 찾아올 위치입니다.
    # shape: (num_points, 3)
    q_coords_ref = Q_space_ref.tabulate_dof_coordinates()[:, :3]

    while t < total_time:
        t += dt

        # (1) 현재 시간에 맞는 XDMF 파일 찾기
        # schedule = {0.0: "file_0.xdmf", 10.0: "file_10.xdmf", ...}
        # 현재 t에 해당하는 각도(파일)를 찾습니다.
        # 예: t=5초이고 1초당 1도 회전이라면 -> 5도 회전된 파일 필요
        # 여기서는 schedule에 있는 시간 중 가장 가까운 과거 시간을 찾습니다.
        times = sorted(schedule.keys())
        current_sched_time = times[0]
        for st in times:
            if st <= t:
                current_sched_time = st
            else:
                break

        current_file = schedule[current_sched_time]

        # 현재 회전 각도 계산 (파일명이나 스케줄 키를 통해 추정 필요)
        # 예: schedule의 key가 '각도'가 아니라 '시간'이라면,
        # 각속도를 알거나 파일명에서 각도를 파싱해야 합니다.
        # 여기서는 간단히: 1초당 10도 회전한다고 가정하거나,
        # schedule 딕셔너리 구조를 {시간: (파일경로, 각도)} 로 받는게 좋습니다.
        # **임시: 파일명에서 각도 추출 (예: ..._10.xdmf -> 10도)**
        try:
            angle_deg = float(current_file.split('_')[-1].replace('.xdmf', ''))
        except:
            angle_deg = 0.0

        angle_rad = np.radians(angle_deg)

        print(f"[HeatSolver] t={t:.2f}s | Reading Q from '{current_file}' (Angle={angle_deg} deg)")

        # (2) 현재 스텝의 전자기장 파일(Rotated Mesh) 열기
        with io.XDMFFile(comm, current_file, "r") as xdmf_rot:
            domain_rot = xdmf_rot.read_mesh(name="mesh")

        # Q Function Space 및 함수 생성
        Q_space_rot = fem.functionspace(domain_rot, ("DG", 0))
        Q_rot = fem.Function(Q_space_rot)

        # [수정] read_function 대신 h5py를 사용하여 HDF5 데이터 직접 읽기
        import h5py
        h5_file = current_file.replace(".xdmf", ".h5")

        try:
            with h5py.File(h5_file, "r") as f:
                # XDMF 구조상 데이터는 "/Function/함수이름/타임스텝" 경로에 저장됨
                # 여기서는 "HeatSource"라는 이름으로 저장했으므로 real/imag로 나뉘어 있을 수 있음

                # 1. 실수부 읽기
                if "Function/real_HeatSource/0" in f:
                    data_real = f["Function/real_HeatSource/0"][:]

                    # 2. 허수부 읽기 (존재할 경우)
                    data_imag = None
                    if "Function/imag_HeatSource/0" in f:
                        data_imag = f["Function/imag_HeatSource/0"][:]

                    # 3. Q_rot에 값 할당
                    # (주의: 병렬 실행 시 메쉬 순서가 섞일 수 있으나, 단일 프로세스에서는 순서가 일치함)
                    flat_real = data_real.flatten()

                    if np.issubdtype(Q_rot.x.array.dtype, np.complexfloating):
                        # 복소수 모드인 경우
                        if data_imag is not None:
                            Q_rot.x.array[:] = flat_real + 1j * data_imag.flatten()
                        else:
                            Q_rot.x.array[:] = flat_real + 0j
                    else:
                        # 실수 모드인 경우
                        Q_rot.x.array[:] = flat_real
                else:
                    print(f"[Warning] 'HeatSource' dataset not found in {h5_file}")
                    Q_rot.x.array[:] = 0.0

        except Exception as e:
            print(f"[Error] Failed to read HDF5 data from {h5_file}: {e}")
            Q_rot.x.array[:] = 0.0

        # (3) [핵심] Reference 좌표를 회전시켜서 Rotated Mesh에서 값 찾기
        # Q_ref(x) = Q_rot(R * x)

        # 회전 행렬 (Y축 기준 회전 예시 - Geo 파일에 따름)
        # Geo 파일: Rotate {{0, 1, 0}, {cx, cy, cz}, angle}
        c, s = np.cos(angle_rad), np.sin(angle_rad)

        # 회전 중심
        cx, cy, cz = rotation_center

        # 좌표 이동 (중심을 원점으로)
        x_shifted = q_coords_ref[:, 0] - cx
        y_shifted = q_coords_ref[:, 1] - cy
        z_shifted = q_coords_ref[:, 2] - cz

        # Y축 회전 적용
        # x' = x cos - z sin
        # z' = x sin + z cos
        x_rot = x_shifted * c - z_shifted * s
        y_rot = y_shifted  # Y축 회전이면 Y는 그대로
        z_rot = x_shifted * s + z_shifted * c

        # 다시 중심으로 복귀
        target_points = np.zeros_like(q_coords_ref)
        target_points[:, 0] = x_rot + cx
        target_points[:, 1] = y_rot + cy
        target_points[:, 2] = z_rot + cz

        # (4) 충돌 감지 및 값 평가 (Point Evaluation)
        # domain_rot(회전된 메쉬)에서 target_points(회전된 좌표)의 값을 찾습니다.

        # BoundingBoxTree 생성
        tree = geometry.bb_tree(domain_rot, domain_rot.topology.dim)

        # 충돌 후보 찾기
        cell_candidates = geometry.compute_collisions_points(tree, target_points)

        # 실제 충돌 셀 찾기
        colliding_cells = geometry.compute_colliding_cells(domain_rot, cell_candidates, target_points)

        # 값 가져오기
        # Q_rot는 DG0(셀 중심 값)이므로, 찾은 셀의 값을 그대로 가져오면 됩니다.

        # Q_external 초기화
        Q_external.x.array[:] = 0.0

        # 각 점에 대해 값 할당
        # (주의: 대량의 점에 대해 파이썬 루프는 느릴 수 있으므로 벡터화가 좋지만,
        #  dolfinx 구조상 안전하게 리스트 처리하거나 C++ 함수 활용 권장)

        # colliding_cells.links(i)는 i번째 점이 포함된 셀의 인덱스 리스트를 반환
        # DG0 공간이므로 점 하나당 셀 하나만 매칭되면 됨.

        # 빠른 처리를 위한 배열 준비
        found_cells = np.full(len(target_points), -1, dtype=np.int32)

        # 이 부분은 C++ 바인딩을 쓰면 더 빠르지만, 파이썬 레벨에서 처리
        for i in range(len(target_points)):
            cells = colliding_cells.links(i)
            if len(cells) > 0:
                found_cells[i] = cells[0]  # 첫 번째 찾은 셀 사용

        # 유효한 셀을 찾은 인덱스 마스크
        valid_mask = found_cells != -1

        # 값 복사: Q_ref[i] = Q_rot[found_cells[i]]
        # Q_rot.x.array는 셀 인덱스 순서대로 값이 들어있음 (DG0)
        if np.any(valid_mask):
            # Q_rot의 값을 가져옴
            source_values = Q_rot.x.array[found_cells[valid_mask]]
            # Q_external(Ref)에 할당
            Q_external.x.array[valid_mask] = source_values

        print(f"  Mapped Q values. Max Q: {np.max(Q_external.x.array):.2f}")

        # (5) 열 방정식 풀기
        problem.solve()

        # (6) 결과 저장
        T_n.x.array[:] = problem.u.x.array[:]
        xdmf_out.write_function(T_n, t)

        if comm.rank == 0:
            print(f"  Step t={t:.2f}s | Max Temp = {np.max(T_n.x.array):.2f} C")

    xdmf_out.close()
    print("[HeatSolver] Done.")


def solve_transient_heat_transfer_static(
        comm,
        static_xdmf_path,
        output_file="result3/temperature_evolution_static.xdmf",
        dt=1.0,
        total_time=10.0,
        initial_temp=25.0
):
    """
    회전 없이 고정된 메쉬(0도)에서 열 해석을 수행하여 가열이 제대로 되는지 검증합니다.
    """
    # 1. 메쉬 로드
    print(f"[HeatSolver] Loading STATIC mesh from {static_xdmf_path}...")
    with io.XDMFFile(comm, static_xdmf_path, "r") as xdmf:
        domain = xdmf.read_mesh(name="mesh")
        cell_tags = xdmf.read_meshtags(domain, name="PhysicalGroups")

    # 2. Function Space 정의
    V = fem.functionspace(domain, ("Lagrange", 1))
    Q_space = fem.functionspace(domain, ("DG", 0))

    # 3. 물성치 설정
    D0 = fem.functionspace(domain, ("DG", 0))
    rho = fem.Function(D0)
    cp = fem.Function(D0)
    k_therm = fem.Function(D0)

    # 기본값 (Air) - 전체를 공기로 초기화
    rho.x.array[:] = 1.2
    cp.x.array[:] = 1000.0
    k_therm.x.array[:] = 0.026

    # ID 찾기 (하드코딩 대신 변수 사용 권장)
    id_glass = 1924
    id_mass = 1925
    id_air = 1930

    def assign_prop(tag, r, c, k):
        cells = cell_tags.find(tag)
        if len(cells) > 0:
            rho.x.array[cells] = r
            cp.x.array[cells] = c
            k_therm.x.array[cells] = k

    # Glass & Mass 물성 할당 (덮어쓰기)
    assign_prop(id_glass, 2500, 840, 1.0)
    assign_prop(id_mass, 1000, 3500, 0.5)

    # 4. 초기 온도 설정
    T_n = fem.Function(V)
    T_n.name = "Temperature"
    T_n.x.array[:] = initial_temp

    # 5. 열원(Q) 로드 (HDF5 직접 읽기 - 한 번만 수행)
    Q_external = fem.Function(Q_space)

    import h5py
    h5_file = static_xdmf_path.replace(".xdmf", ".h5")

    try:
        with h5py.File(h5_file, "r") as f:
            # 실수부 읽기
            if "Function/real_HeatSource/0" in f:
                data_real = f["Function/real_HeatSource/0"][:]

                # 허수부 읽기
                data_imag = None
                if "Function/imag_HeatSource/0" in f:
                    data_imag = f["Function/imag_HeatSource/0"][:]

                flat_real = data_real.flatten()

                if np.issubdtype(Q_external.x.array.dtype, np.complexfloating):
                    if data_imag is not None:
                        Q_external.x.array[:] = flat_real + 1j * data_imag.flatten()
                    else:
                        Q_external.x.array[:] = flat_real + 0j
                else:
                    Q_external.x.array[:] = flat_real
            else:
                print(f"[Warning] 'HeatSource' dataset not found in {h5_file}")
                Q_external.x.array[:] = 0.0
    except Exception as e:
        print(f"[Error] Failed to read HDF5 data: {e}")
        Q_external.x.array[:] = 0.0

    # [노이즈 제거] 공기/유리 영역 강제 0 (매우 중요)
    air_cells = cell_tags.find(id_air)
    glass_cells = cell_tags.find(id_glass)
    if len(air_cells) > 0: Q_external.x.array[air_cells] = 0.0
    if len(glass_cells) > 0: Q_external.x.array[glass_cells] = 0.0

    print(f"  Loaded Q values. Max Q: {np.max(Q_external.x.array):.2f}")

    # 6. 열 방정식 정의
    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # ufl.inner 사용 (Complex 모드 호환)
    term_time = (rho * cp / dt) * ufl.inner(T - T_n, v) * ufl.dx
    term_diff = k_therm * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx
    term_source = ufl.inner(Q_external, v) * ufl.dx

    F = term_time + term_diff - term_source

    petsc_options_prefix_heat = "pc_heat_static"
    problem = fem.petsc.LinearProblem(ufl.lhs(F), ufl.rhs(F), bcs=[],
                                      petsc_options={"ksp_type": "cg", "pc_type": "gamg"},
                                      petsc_options_prefix=petsc_options_prefix_heat)

    # 7. 결과 저장 및 시간 루프
    xdmf_out = io.XDMFFile(comm, output_file, "w")
    xdmf_out.write_mesh(domain)
    xdmf_out.write_function(T_n, 0.0)

    t = 0.0
    print(f"[HeatSolver] Starting STATIC simulation for {total_time}s...")

    while t < total_time:
        t += dt
        problem.solve()
        T_n.x.array[:] = problem.u.x.array[:]
        xdmf_out.write_function(T_n, t)

        if comm.rank == 0:
            print(f"  Step t={t:.2f}s | Max Temp = {np.max(T_n.x.array):.2f} C")

    xdmf_out.close()
    print("[HeatSolver] Static simulation done.")


def solve_transient_heat_transfer_static2(
        comm,
        static_xdmf_path,
        output_file="result3/temperature_evolution_static.xdmf",
        dt=1.0,
        total_time=10.0,
        initial_temp=25.0
):
    """
    회전 없이 고정된 메쉬(0도)에서 열 해석을 수행하여 가열이 제대로 되는지 검증합니다.
    (시각화용 변수를 분리하여 Air/Glass 영역을 0도로 표시하여 음식만 보이게 합니다)
    """
    # 1. 메쉬 로드
    print(f"[HeatSolver] Loading STATIC mesh from {static_xdmf_path}...")
    with io.XDMFFile(comm, static_xdmf_path, "r") as xdmf:
        domain = xdmf.read_mesh(name="mesh")
        cell_tags = xdmf.read_meshtags(domain, name="PhysicalGroups")

    # 2. Function Space 정의
    V = fem.functionspace(domain, ("Lagrange", 1))
    Q_space = fem.functionspace(domain, ("DG", 0))

    # 3. 물성치 설정
    D0 = fem.functionspace(domain, ("DG", 0))
    rho = fem.Function(D0)
    cp = fem.Function(D0)
    k_therm = fem.Function(D0)

    # 기본값 (Air) - 전체를 공기로 초기화
    rho.x.array[:] = 1.2
    cp.x.array[:] = 1000.0
    k_therm.x.array[:] = 0.026

    # ID 찾기
    id_glass = 1924
    id_mass = 1925
    id_air = 1930
    ids_mass_list = [1925]

    def assign_prop(tag, r, c, k):
        cells = cell_tags.find(tag)
        if len(cells) > 0:
            rho.x.array[cells] = r
            cp.x.array[cells] = c
            k_therm.x.array[cells] = k

    # Glass & Mass 물성 할당 (덮어쓰기)
    assign_prop(id_glass, 2500, 840, 1.0)
    assign_prop(id_mass, 1000, 3500, 0.5)

    # 4. 초기 온도 설정 (계산용)
    T_n = fem.Function(V)
    T_n.name = "Temperature"
    T_n.x.array[:] = initial_temp

    # [추가] 시각화 저장용 변수 (계산에 영향 안 줌)
    T_vis = fem.Function(V)
    T_vis.name = "Temperature"
    T_vis.x.array[:] = initial_temp

    # 5. 열원(Q) 로드 (HDF5 직접 읽기 - 한 번만 수행)
    Q_external = fem.Function(Q_space)

    import h5py
    h5_file = static_xdmf_path.replace(".xdmf", ".h5")

    try:
        with h5py.File(h5_file, "r") as f:
            # 실수부 읽기
            if "Function/real_HeatSource/0" in f:
                data_real = f["Function/real_HeatSource/0"][:]

                # 허수부 읽기
                data_imag = None
                if "Function/imag_HeatSource/0" in f:
                    data_imag = f["Function/imag_HeatSource/0"][:]

                flat_real = data_real.flatten()

                if np.issubdtype(Q_external.x.array.dtype, np.complexfloating):
                    if data_imag is not None:
                        Q_external.x.array[:] = flat_real + 1j * data_imag.flatten()
                    else:
                        Q_external.x.array[:] = flat_real + 0j
                else:
                    Q_external.x.array[:] = flat_real
            else:
                print(f"[Warning] 'HeatSource' dataset not found in {h5_file}")
                Q_external.x.array[:] = 0.0
    except Exception as e:
        print(f"[Error] Failed to read HDF5 data: {e}")
        Q_external.x.array[:] = 0.0

    # [노이즈 제거] 공기/유리 영역 강제 0 (매우 중요)
    air_cells = cell_tags.find(id_air)
    glass_cells = cell_tags.find(id_glass)
    if len(air_cells) > 0: Q_external.x.array[air_cells] = 0.0
    if len(glass_cells) > 0: Q_external.x.array[glass_cells] = 0.0

    print(f"  Loaded Q values. Max Q: {np.max(Q_external.x.array):.2f}")

    # 6. 열 방정식 정의
    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # ufl.inner 사용 (Complex 모드 호환)
    term_time = (rho * cp / dt) * ufl.inner(T - T_n, v) * ufl.dx
    term_diff = k_therm * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx
    term_source = ufl.inner(Q_external, v) * ufl.dx

    F = term_time + term_diff - term_source

    petsc_options_prefix_heat = "pc_heat_static"
    problem = fem.petsc.LinearProblem(ufl.lhs(F), ufl.rhs(F), bcs=[],
                                      petsc_options={"ksp_type": "cg", "pc_type": "gamg"},
                                      petsc_options_prefix=petsc_options_prefix_heat)

    # 7. 결과 저장 및 시간 루프
    xdmf_out = io.XDMFFile(comm, output_file, "w")
    xdmf_out.write_mesh(domain)

    # [추가] ParaView 필터링을 위해 태그 정보도 저장
    cell_tags.name = "PhysicalGroups"
    xdmf_out.write_meshtags(cell_tags, domain.geometry)

    # -------------------------------------------------------------------------
    # [시각화용] Mass가 아닌 영역의 노드(DOF) 찾기
    # -------------------------------------------------------------------------
    mass_cells_vis = np.array([], dtype=np.int32)
    for tag in ids_mass_list:
        found = cell_tags.find(tag)
        if len(found) > 0:
            mass_cells_vis = np.concatenate((mass_cells_vis, found))

    # Connectivity 계산 (locate_dofs_topological 에러 방지)
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)

    mass_dofs = fem.locate_dofs_topological(V, domain.topology.dim, mass_cells_vis)
    mass_dofs = np.unique(mass_dofs)

    all_dofs_indices = np.arange(len(T_n.x.array), dtype=np.int32)
    dofs_to_zero = np.setdiff1d(all_dofs_indices, mass_dofs)

    # 초기 상태 저장 (시각화용 변수 사용)
    if len(dofs_to_zero) > 0:
        T_vis.x.array[dofs_to_zero] = 0.0

    xdmf_out.write_function(T_vis, 0.0)

    t = 0.0
    print(f"[HeatSolver] Starting STATIC simulation for {total_time}s...")

    while t < total_time:
        t += dt
        problem.solve()

        # [중요] 계산 결과는 T_n에 저장 (물리적 값 유지)
        T_n.x.array[:] = problem.u.x.array[:]

        # [시각화 트릭] T_vis에 복사 후 Mass가 아닌 곳만 0으로 변경
        T_vis.x.array[:] = T_n.x.array[:]
        if len(dofs_to_zero) > 0:
            T_vis.x.array[dofs_to_zero] = 0.0

        xdmf_out.write_function(T_vis, t)

        if comm.rank == 0:
            print(f"  Step t={t:.2f}s | Max Temp (Mass) = {np.max(T_n.x.array):.2f} C")

    xdmf_out.close()
    print("[HeatSolver] Static simulation done.")

def solve_transient_heat_transfer_static3(
        comm,
        static_xdmf_path,
        output_file="result3/temperature_evolution_static.xdmf",
        dt=1.0,
        total_time=10.0,
        initial_temp=25.0
):
    """
    회전 없이 고정된 메쉬(0도)에서 열 해석을 수행합니다.
    * output_file: 원본 결과 (전체 도메인 온도)
    * output_file_noair: 시각화용 결과 (Air/Glass = 0)
    """
    # 1. 메쉬 로드
    print(f"[HeatSolver] Loading STATIC mesh from {static_xdmf_path}...")
    with io.XDMFFile(comm, static_xdmf_path, "r") as xdmf:
        domain = xdmf.read_mesh(name="mesh")
        cell_tags = xdmf.read_meshtags(domain, name="PhysicalGroups")

    # 2. Function Space 정의
    V = fem.functionspace(domain, ("Lagrange", 1))
    Q_space = fem.functionspace(domain, ("DG", 0))

    # 3. 물성치 설정
    D0 = fem.functionspace(domain, ("DG", 0))
    rho = fem.Function(D0)
    cp = fem.Function(D0)
    k_therm = fem.Function(D0)

    # 기본값 (Air)
    rho.x.array[:] = 1.2
    cp.x.array[:] = 1000.0
    k_therm.x.array[:] = 0.026

    # ID 찾기
    id_glass = 1924
    id_mass = 1925
    id_air = 1930
    ids_mass_list = [1925]

    def assign_prop(tag, r, c, k):
        cells = cell_tags.find(tag)
        if len(cells) > 0:
            rho.x.array[cells] = r
            cp.x.array[cells] = c
            k_therm.x.array[cells] = k

    assign_prop(id_glass, 2500, 840, 1.0)
    assign_prop(id_mass, 1000, 3500, 0.5)

    # 4. 초기 온도 설정 (계산용)
    T_n = fem.Function(V)
    T_n.name = "Temperature"
    T_n.x.array[:] = initial_temp

    # [추가] 시각화 저장용 변수
    T_vis = fem.Function(V)
    T_vis.name = "Temperature"
    T_vis.x.array[:] = initial_temp

    # 5. 열원(Q) 로드
    Q_external = fem.Function(Q_space)
    h5_file = static_xdmf_path.replace(".xdmf", ".h5")

    try:
        with h5py.File(h5_file, "r") as f:
            if "Function/real_HeatSource/0" in f:
                data_real = f["Function/real_HeatSource/0"][:]
                data_imag = None
                if "Function/imag_HeatSource/0" in f:
                    data_imag = f["Function/imag_HeatSource/0"][:]
                flat_real = data_real.flatten()
                if np.issubdtype(Q_external.x.array.dtype, np.complexfloating):
                    if data_imag is not None:
                        Q_external.x.array[:] = flat_real + 1j * data_imag.flatten()
                    else:
                        Q_external.x.array[:] = flat_real + 0j
                else:
                    Q_external.x.array[:] = flat_real
            else:
                print(f"[Warning] 'HeatSource' dataset not found in {h5_file}")
                Q_external.x.array[:] = 0.0
    except Exception as e:
        print(f"[Error] Failed to read HDF5 data: {e}")
        Q_external.x.array[:] = 0.0

    # [노이즈 제거] 공기/유리 영역 강제 0
    air_cells = cell_tags.find(id_air)
    glass_cells = cell_tags.find(id_glass)
    if len(air_cells) > 0: Q_external.x.array[air_cells] = 0.0
    if len(glass_cells) > 0: Q_external.x.array[glass_cells] = 0.0

    print(f"  Loaded Q values. Max Q: {np.max(Q_external.x.array):.2f}")

    # 6. 열 방정식 정의
    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    term_time = (rho * cp / dt) * ufl.inner(T - T_n, v) * ufl.dx
    term_diff = k_therm * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx
    term_source = ufl.inner(Q_external, v) * ufl.dx
    F = term_time + term_diff - term_source

    petsc_options_prefix_heat = "pc_heat_static"
    problem = fem.petsc.LinearProblem(ufl.lhs(F), ufl.rhs(F), bcs=[],
                                      petsc_options={"ksp_type": "cg", "pc_type": "gamg"},
                                      petsc_options_prefix=petsc_options_prefix_heat)

    # -------------------------------------------------------------------------
    # 7. 결과 저장 설정 (두 개의 파일 생성)
    # -------------------------------------------------------------------------
    # 파일명 생성: test.xdmf -> test_noair.xdmf
    base, ext = os.path.splitext(output_file)
    output_file_noair = f"{base}_noair{ext}"

    # (1) 원본 파일 (전체 데이터)
    xdmf_out = io.XDMFFile(comm, output_file, "w")
    xdmf_out.write_mesh(domain)
    cell_tags.name = "PhysicalGroups"
    xdmf_out.write_meshtags(cell_tags, domain.geometry)
    xdmf_out.write_function(T_n, 0.0)

    # (2) 시각화용 파일 (No Air)
    xdmf_out_noair = io.XDMFFile(comm, output_file_noair, "w")
    xdmf_out_noair.write_mesh(domain)
    xdmf_out_noair.write_meshtags(cell_tags, domain.geometry)

    # -------------------------------------------------------------------------
    # [시각화용] Mass가 아닌 영역의 노드(DOF) 찾기
    # -------------------------------------------------------------------------
    mass_cells_vis = np.array([], dtype=np.int32)
    for tag in ids_mass_list:
        found = cell_tags.find(tag)
        if len(found) > 0:
            mass_cells_vis = np.concatenate((mass_cells_vis, found))

    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)
    mass_dofs = fem.locate_dofs_topological(V, domain.topology.dim, mass_cells_vis)
    mass_dofs = np.unique(mass_dofs)
    all_dofs_indices = np.arange(len(T_n.x.array), dtype=np.int32)
    dofs_to_zero = np.setdiff1d(all_dofs_indices, mass_dofs)

    # 초기 상태 저장 (No Air)
    if len(dofs_to_zero) > 0:
        T_vis.x.array[dofs_to_zero] = 0.0
    xdmf_out_noair.write_function(T_vis, 0.0)

    t = 0.0
    print(f"[HeatSolver] Starting STATIC simulation for {total_time}s...")
    print(f"[HeatSolver] Outputs: '{output_file}' (Full), '{output_file_noair}' (Mass Only)")

    while t < total_time:
        t += dt
        problem.solve()

        # [중요] 계산 결과는 T_n에 저장 (물리적 값 유지)
        T_n.x.array[:] = problem.u.x.array[:]

        # (1) 원본 저장
        xdmf_out.write_function(T_n, t)

        # (2) 시각화용 저장 (복사 후 0으로 변경)
        T_vis.x.array[:] = T_n.x.array[:]
        if len(dofs_to_zero) > 0:
            T_vis.x.array[dofs_to_zero] = 0.0
        xdmf_out_noair.write_function(T_vis, t)

        if comm.rank == 0:
            print(f"  Step t={t:.2f}s | Max Temp (Mass) = {np.max(T_n.x.array):.2f} C")

    xdmf_out.close()
    xdmf_out_noair.close()
    print("[HeatSolver] Static simulation done.")

def solve_transient_heat_transfer_rotating2(
        comm,
        schedule,
        rotation_center=None,
        output_file="result3/temperature_evolution.xdmf",
        dt=1.0,
        total_time=10.0,
        initial_temp=25.0
):
    """
    회전하는 메쉬의 Q 데이터를 고정된 메쉬(t=0)로 매핑하여 열 해석을 수행합니다.
    (Static 해석과 동일한 노이즈 제거 로직 적용)
    """

    # 1. 기준 메쉬(Reference Mesh) 로드 - t=0 (0도) 파일 사용

    print(f"[HeatSolver] Loading REFERENCE mesh from {ref_file}...")

    with io.XDMFFile(comm, ref_file, "r") as xdmf:
        domain_ref = xdmf.read_mesh(name="mesh")
        cell_tags_ref = xdmf.read_meshtags(domain_ref, name="PhysicalGroups")

    # -------------------------------------------------------------------------
    # [설정] ID 정의 (Static 함수와 동일하게 설정)
    # -------------------------------------------------------------------------
    id_glass = 1924
    id_mass = 1925
    id_air = 1930
    ids_mass_list = [1925]  # Mass가 여러 개일 경우를 대비

    # -------------------------------------------------------------------------
    # [추가] 회전 중심 자동 계산 로직
    # -------------------------------------------------------------------------
    if rotation_center is None:
        glass_cells = cell_tags_ref.find(id_glass)
        if len(glass_cells) > 0:
            domain_ref.topology.create_connectivity(domain_ref.topology.dim, 0)
            vertex_indices = mesh.entities_to_geometry(domain_ref, domain_ref.topology.dim, glass_cells, True)
            unique_vertices = np.unique(vertex_indices)
            coords = domain_ref.geometry.x[unique_vertices]
            min_pt = np.min(coords, axis=0)
            max_pt = np.max(coords, axis=0)
            rotation_center = (min_pt + max_pt) / 2.0
            rotation_center = rotation_center[:3]
            if comm.rank == 0:
                print(f"[HeatSolver] Auto-calculated Rotation Center (Glass): {rotation_center}")
        else:
            rotation_center = (0.155, 0.13, 0.155)
            if comm.rank == 0:
                print("[HeatSolver] Warning: Glass domain not found. Using default rotation center.")

    # 2. Function Space 정의
    V_ref = fem.functionspace(domain_ref, ("Lagrange", 1))
    Q_space_ref = fem.functionspace(domain_ref, ("DG", 0))

    # 3. 물성치 설정
    D0 = fem.functionspace(domain_ref, ("DG", 0))
    rho = fem.Function(D0);
    rho.x.array[:] = 1.2
    cp = fem.Function(D0);
    cp.x.array[:] = 1000.0
    k_therm = fem.Function(D0);
    k_therm.x.array[:] = 0.026

    def assign_prop(tag, r, c, k):
        cells = cell_tags_ref.find(tag)
        if len(cells) > 0:
            rho.x.array[cells] = r
            cp.x.array[cells] = c
            k_therm.x.array[cells] = k

    assign_prop(id_glass, 2500, 840, 1.0)
    assign_prop(id_mass, 1000, 3500, 0.5)

    # 초기 온도
    T_n = fem.Function(V_ref)
    T_n.name = "Temperature"
    T_n.x.array[:] = initial_temp

    # 4. 열 방정식 정의
    T = ufl.TrialFunction(V_ref)
    v = ufl.TestFunction(V_ref)
    Q_external = fem.Function(Q_space_ref)

    term_time = (rho * cp / dt) * ufl.inner(T - T_n, v) * ufl.dx
    term_diff = k_therm * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx
    term_source = ufl.inner(Q_external, v) * ufl.dx

    F = term_time + term_diff - term_source

    petsc_options_prefix_heat = "pc_heat"
    problem = fem.petsc.LinearProblem(ufl.lhs(F), ufl.rhs(F), bcs=[],
                                      petsc_options={"ksp_type": "cg", "pc_type": "gamg"},
                                      petsc_options_prefix=petsc_options_prefix_heat)

    # 5. 결과 저장용 파일
    xdmf_out = io.XDMFFile(comm, output_file, "w")
    xdmf_out.write_mesh(domain_ref)
    xdmf_out.write_function(T_n, 0.0)

    # 6. 시간 루프
    t = 0.0
    print(f"[HeatSolver] Starting simulation...")

    q_coords_ref = Q_space_ref.tabulate_dof_coordinates()[:, :3]

    # [중요] Mass 셀 인덱스 미리 찾기 (노이즈 제거용)
    all_indices = np.arange(len(Q_external.x.array), dtype=np.int32)
    mass_cells = np.array([], dtype=np.int32)
    for tag in ids_mass_list:
        found = cell_tags_ref.find(tag)
        if len(found) > 0:
            mass_cells = np.concatenate((mass_cells, found))

    # Mass가 아닌 모든 셀 (차집합)
    non_mass_cells = np.setdiff1d(all_indices, mass_cells)

    while t < total_time:
        t += dt

        # (1) 스케줄링
        times = sorted(schedule.keys())
        current_sched_time = times[0]
        for st in times:
            if st <= t:
                current_sched_time = st
            else:
                break
        current_file = schedule[current_sched_time]

        try:
            angle_deg = float(current_file.split('_')[-1].replace('.xdmf', ''))
        except:
            angle_deg = 0.0
        angle_rad = np.radians(angle_deg)

        print(f"[HeatSolver] t={t:.2f}s | Reading Q from '{current_file}' (Angle={angle_deg} deg)")

        # (2) HDF5 데이터 읽기
        with io.XDMFFile(comm, current_file, "r") as xdmf_rot:
            domain_rot = xdmf_rot.read_mesh(name="mesh")

        Q_space_rot = fem.functionspace(domain_rot, ("DG", 0))
        Q_rot = fem.Function(Q_space_rot)

        import h5py
        h5_file = current_file.replace(".xdmf", ".h5")
        try:
            with h5py.File(h5_file, "r") as f:
                if "Function/real_HeatSource/0" in f:
                    data_real = f["Function/real_HeatSource/0"][:]
                    data_imag = None
                    if "Function/imag_HeatSource/0" in f:
                        data_imag = f["Function/imag_HeatSource/0"][:]

                    flat_real = data_real.flatten()
                    if np.issubdtype(Q_rot.x.array.dtype, np.complexfloating):
                        if data_imag is not None:
                            Q_rot.x.array[:] = flat_real + 1j * data_imag.flatten()
                        else:
                            Q_rot.x.array[:] = flat_real + 0j
                    else:
                        Q_rot.x.array[:] = flat_real
                else:
                    Q_rot.x.array[:] = 0.0
        except Exception as e:
            print(f"[Error] HDF5 read failed: {e}")
            Q_rot.x.array[:] = 0.0

        # (3) 회전 매핑
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy, cz = rotation_center

        x_shifted = q_coords_ref[:, 0] - cx
        y_shifted = q_coords_ref[:, 1] - cy
        z_shifted = q_coords_ref[:, 2] - cz

        x_rot = x_shifted * c - z_shifted * s
        y_rot = y_shifted
        z_rot = x_shifted * s + z_shifted * c

        target_points = np.zeros_like(q_coords_ref)
        target_points[:, 0] = x_rot + cx
        target_points[:, 1] = y_rot + cy
        target_points[:, 2] = z_rot + cz

        # (4) 충돌 감지 및 값 할당
        tree = geometry.bb_tree(domain_rot, domain_rot.topology.dim)
        cell_candidates = geometry.compute_collisions_points(tree, target_points)
        colliding_cells = geometry.compute_colliding_cells(domain_rot, cell_candidates, target_points)

        Q_external.x.array[:] = 0.0
        found_cells = np.full(len(target_points), -1, dtype=np.int32)

        for i in range(len(target_points)):
            cells = colliding_cells.links(i)
            if len(cells) > 0:
                found_cells[i] = cells[0]

        valid_mask = found_cells != -1
        if np.any(valid_mask):
            source_values = Q_rot.x.array[found_cells[valid_mask]]
            Q_external.x.array[valid_mask] = source_values

        # -------------------------------------------------------------------------
        # [중요] 음식(Mass)을 제외한 모든 영역의 발열량 강제 제거 (Static과 동일 로직)
        # -------------------------------------------------------------------------
        if len(non_mass_cells) > 0:
            Q_external.x.array[non_mass_cells] = 0.0

        print(f"  Mapped Q values. Max Q in Mass: {np.max(Q_external.x.array):.2f}")

        # (5) 풀이 및 저장
        problem.solve()
        T_n.x.array[:] = problem.u.x.array[:]
        xdmf_out.write_function(T_n, t)

        if comm.rank == 0:
            print(f"  Step t={t:.2f}s | Max Temp = {np.max(T_n.x.array):.2f} C")

    xdmf_out.close()
    print("[HeatSolver] Done.")


def analyze_heating_uniformity(comm, schedule, rotation_center):
    """
    저장된 XDMF/H5 파일들의 Heat Source(Q)를 로드하고 매핑하여,
    회전 시 가열 균일도가 얼마나 개선되는지 수치적으로 분석합니다.
    """
    import h5py

    # 1. 기준 메쉬(Reference Mesh, 0도) 로드
    ref_time = 0.0
    if ref_time not in schedule:
        ref_time = min(schedule.keys())

    print(f"\n[Analysis] Loading REFERENCE mesh from {ref_file}...")

    with io.XDMFFile(comm, ref_file, "r") as xdmf:
        domain_ref = xdmf.read_mesh(name="mesh")
        cell_tags_ref = xdmf.read_meshtags(domain_ref, name="PhysicalGroups")

    # Mass(음식) 영역 식별
    id_mass = 1925  # (주의: 실제 ID 확인 필요)
    mass_cells = cell_tags_ref.find(id_mass)

    if len(mass_cells) == 0:
        print("[Analysis] Error: Mass domain not found!")
        return

    # Q 값을 담을 공간 (DG0)
    Q_space_ref = fem.functionspace(domain_ref, ("DG", 0))

    # 기준 메쉬의 좌표점 (DG0이므로 셀 중심점)
    q_coords_ref = Q_space_ref.tabulate_dof_coordinates()[:, :3]

    # Mass 영역에 해당하는 DOF 인덱스만 추출 (통계 낼 때 공기/유리 제외하기 위함)
    # DG0에서는 cell index와 dof index가 1:1 대응 (일반적으로)
    mass_dofs = mass_cells

    print(f"[Analysis] Analyzing uniformity on {len(mass_dofs)} mass cells...")

    # -------------------------------------------------------------------------
    # 모든 각도의 Q 데이터를 Reference 좌표계로 매핑하여 저장
    # -------------------------------------------------------------------------
    # q_stack: [각도 개수, 전체 셀 개수]
    num_angles = len(schedule)
    num_cells = len(q_coords_ref)
    q_stack = np.zeros((num_angles, num_cells))

    sorted_times = sorted(schedule.keys())

    for idx, t in enumerate(sorted_times):
        current_file = schedule[t]

        # 파일명에서 각도 추출 (예: ..._10.xdmf -> 10.0)
        try:
            angle_deg = float(current_file.split('_')[-1].replace('.xdmf', ''))
        except:
            angle_deg = 0.0

        angle_rad = np.radians(angle_deg)

        if comm.rank == 0:
            print(f"  - Processing Angle {angle_deg:>5.1f} deg ({idx + 1}/{num_angles})", end="\r")

        # 1) 현재 각도의 메쉬 및 데이터 로드
        with io.XDMFFile(comm, current_file, "r") as xdmf_rot:
            domain_rot = xdmf_rot.read_mesh(name="mesh")

        # HDF5에서 Q값 읽기
        h5_file = current_file.replace(".xdmf", ".h5")
        q_values_rot = np.zeros(num_cells)  # 크기는 같다고 가정

        try:
            with h5py.File(h5_file, "r") as f:
                if "Function/real_HeatSource/0" in f:
                    data = f["Function/real_HeatSource/0"][:]
                    q_values_rot = data.flatten()
        except Exception as e:
            pass

        # 2) 좌표 회전 및 매핑 (Ref -> Rot)
        # Q_ref(x) = Q_rot(R * x)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy, cz = rotation_center

        x_shifted = q_coords_ref[:, 0] - cx
        y_shifted = q_coords_ref[:, 1] - cy
        z_shifted = q_coords_ref[:, 2] - cz

        # Y축 회전
        x_rot = x_shifted * c - z_shifted * s
        y_rot = y_shifted
        z_rot = x_shifted * s + z_shifted * c

        target_points = np.zeros_like(q_coords_ref)
        target_points[:, 0] = x_rot + cx
        target_points[:, 1] = y_rot + cy
        target_points[:, 2] = z_rot + cz

        # 3) 충돌 감지 및 값 가져오기
        tree = geometry.bb_tree(domain_rot, domain_rot.topology.dim)
        cell_candidates = geometry.compute_collisions_points(tree, target_points)
        colliding_cells = geometry.compute_colliding_cells(domain_rot, cell_candidates, target_points)

        # 매핑된 값 저장
        mapped_q = np.zeros(num_cells)

        # (성능을 위해 단순 루프 사용, C++ 바인딩 권장되나 여기선 파이썬으로 처리)
        # Mass 영역만 매핑해도 되지만, 전체 매핑 후 나중에 필터링
        for i in range(len(target_points)):
            cells = colliding_cells.links(i)
            if len(cells) > 0:
                mapped_q[i] = q_values_rot[cells[0]]

        q_stack[idx, :] = mapped_q

    print(f"\n[Analysis] Data loading and mapping complete.")

    # -------------------------------------------------------------------------
    # 통계 분석 (Mass 영역만 대상)
    # -------------------------------------------------------------------------
    # q_stack_mass: [각도 개수, Mass 셀 개수]
    q_stack_mass = q_stack[:, mass_dofs]

    # 1. 정지 상태 (0도)
    q_static = q_stack_mass[0, :]
    mean_static = np.mean(q_static)
    std_static = np.std(q_static)
    cov_static = std_static / mean_static if mean_static > 0 else 0

    # 2. 회전 상태 (모든 각도 평균)
    q_rotating = np.mean(q_stack_mass, axis=0)
    mean_rot = np.mean(q_rotating)
    std_rot = np.std(q_rotating)
    cov_rot = std_rot / mean_rot if mean_rot > 0 else 0

    # 3. 최적의 단일 각도 찾기 (재미 삼아)
    covs = []
    for i in range(num_angles):
        q_i = q_stack_mass[i, :]
        m = np.mean(q_i)
        s = np.std(q_i)
        covs.append(s / m if m > 0 else 1e6)

    best_angle_idx = np.argmin(covs)
    best_angle_time = sorted_times[best_angle_idx]
    best_cov = covs[best_angle_idx]

    print("\n" + "=" * 60)
    print("   HEATING UNIFORMITY ANALYSIS REPORT")
    print("=" * 60)
    print(f"Metric: Coefficient of Variation (CoV) = StdDev / Mean")
    print(f"(Lower is Better, 0.0 is perfectly uniform)")
    print("-" * 60)
    print(f"1. Stationary (0 deg):")
    print(f"   - Mean Q: {mean_static:.2f}")
    print(f"   - CoV   : \033[91m{cov_static:.4f}\033[0m (Baseline)")
    print("-" * 60)
    print(f"2. Full Rotation (Averaged over 360 deg):")
    print(f"   - Mean Q: {mean_rot:.2f}")
    print(f"   - CoV   : \033[96m{cov_rot:.4f}\033[0m")

    improvement = (cov_static - cov_rot) / cov_static * 100
    print(f"   => Uniformity Improved by: \033[92m{improvement:.1f}%\033[0m")
    print("-" * 60)
    print(f"3. Best Single Angle (Static):")
    print(f"   - Angle Index: {best_angle_idx} (Time/Angle key: {best_angle_time})")
    print(f"   - CoV        : {best_cov:.4f}")
    print("=" * 60 + "\n")

    return q_rotating  # 평균화된 Q 필드 반환 (필요시 저장 가능)


def analyze_heating_uniformity2(comm, schedule, rotation_center):
    """
    [수정됨]
    1. 'Standard 360 Rotation' (0~350도 모든 파일 평균)을 Baseline으로 설정합니다.
    2. 입력받은 'schedule' (최적화된 조합)이 Baseline보다 얼마나 균일한지(CoV) 비교합니다.
    """
    import h5py
    import glob

    # -------------------------------------------------------------------------
    # 1. 기준 메쉬(Reference Mesh) 및 Mass 영역 설정
    # -------------------------------------------------------------------------
    # 스케줄의 첫 번째 파일로 메쉬 구조 파악
    first_file = list(schedule.values())[0]

    if comm.rank == 0:
        print(f"\n[Analysis] Setting up Reference Mesh from {first_file}...")

    with io.XDMFFile(comm, first_file, "r") as xdmf:
        domain_ref = xdmf.read_mesh(name="mesh")
        cell_tags_ref = xdmf.read_meshtags(domain_ref, name="PhysicalGroups")

    # Mass(음식) 영역 식별
    id_mass = 1925
    mass_cells = cell_tags_ref.find(id_mass)

    if len(mass_cells) == 0:
        if comm.rank == 0: print("[Analysis] Error: Mass domain not found!")
        return

    # Q 값을 담을 공간 (DG0)
    Q_space_ref = fem.functionspace(domain_ref, ("DG", 0))
    q_coords_ref = Q_space_ref.tabulate_dof_coordinates()[:, :3]

    # Mass 영역의 DOF 인덱스
    mass_dofs = mass_cells
    num_cells = len(q_coords_ref)

    # -------------------------------------------------------------------------
    # 2. 데이터 로딩 및 매핑 (캐싱)
    # -------------------------------------------------------------------------
    # 목적: 0도~350도 모든 파일의 데이터를 메모리에 한 번만 로드해둡니다.
    # 그래야 '360도 회전(Baseline)'과 '최적화 스케줄(Target)'을 둘 다 계산할 수 있습니다.

    # 파일명 패턴 파악 (예: ..._10.xdmf)
    # 0도부터 350도까지 파일 경로 리스트 생성
    base_dir = os.path.dirname(first_file)
    base_name = os.path.basename(first_file)
    # "microwave..._10.xdmf" -> prefix 분리
    prefix = base_name.rsplit('_', 1)[0]

    # 0~350도 파일 경로 생성
    all_angle_files = {}
    for ang in range(0, 360, 10):
        fname = os.path.join(base_dir, f"{prefix}_{ang}.xdmf")
        if os.path.exists(fname):
            all_angle_files[ang] = fname

    if comm.rank == 0:
        print(f"[Analysis] Found {len(all_angle_files)} angle files for Baseline calculation.")

    # 데이터 캐시: { 각도(int) : Q_array(numpy) }
    q_cache = {}

    # 로딩 루프
    for ang, fname in all_angle_files.items():
        # HDF5 읽기
        h5_file = fname.replace(".xdmf", ".h5")
        q_vals_rot = np.zeros(num_cells)

        try:
            with h5py.File(h5_file, "r") as f:
                if "Function/real_HeatSource/0" in f:
                    q_vals_rot = f["Function/real_HeatSource/0"][:].flatten()
        except:
            pass  # 파일 없거나 에러나면 0으로 처리

        # 좌표 회전 및 매핑 (Ref 좌표계로 변환)
        angle_rad = np.radians(ang)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy, cz = rotation_center

        x_sh = q_coords_ref[:, 0] - cx
        y_sh = q_coords_ref[:, 1] - cy
        z_sh = q_coords_ref[:, 2] - cz

        # Y축 회전
        x_rot = x_sh * c - z_sh * s
        y_rot = y_shifted = y_sh
        z_rot = x_sh * s + z_sh * c

        target_points = np.zeros_like(q_coords_ref)
        target_points[:, 0] = x_rot + cx
        target_points[:, 1] = y_rot + cy
        target_points[:, 2] = z_rot + cz

        # 충돌 감지 (간이 방식: 메쉬 로드 없이 좌표 변환만 수행했다고 가정하고,
        # 실제로는 정확성을 위해 해당 각도의 메쉬를 로드해서 매핑해야 함.
        # 여기서는 속도를 위해, 그리고 메쉬 토폴로지가 회전만 했다고 가정하고
        # 0도 메쉬의 bb_tree를 사용하여 역매핑을 시도하거나,
        # 위에서 로드한 domain_ref를 회전시켜서 찾습니다.)

        # *정확한 매핑을 위해 해당 각도의 메쉬를 잠시 로드합니다*
        with io.XDMFFile(comm, fname, "r") as xdmf_rot:
            domain_rot = xdmf_rot.read_mesh(name="mesh")

        tree = geometry.bb_tree(domain_rot, domain_rot.topology.dim)
        cell_candidates = geometry.compute_collisions_points(tree, target_points)
        colliding_cells = geometry.compute_colliding_cells(domain_rot, cell_candidates, target_points)

        mapped_q = np.zeros(num_cells)
        # Mass 영역만 매핑
        for i in mass_dofs:
            links = colliding_cells.links(i)
            if len(links) > 0:
                mapped_q[i] = q_vals_rot[links[0]]

        q_cache[ang] = mapped_q

        if comm.rank == 0:
            print(f"  - Loaded & Mapped Angle {ang} deg", end="\r")

    if comm.rank == 0:
        print(f"\n[Analysis] Data caching complete. Calculating statistics...")

    # -------------------------------------------------------------------------
    # 3. 통계 계산 및 비교
    # -------------------------------------------------------------------------

    def calc_stats(q_arrays_list):
        """ Q 배열들의 리스트를 받아 평균 필드의 CoV를 계산 """
        if not q_arrays_list: return 0.0, 0.0, float('inf')

        # 1. 모든 순간의 Q를 합쳐서 평균 Q 필드 생성 (시간 평균)
        q_avg_field = np.mean(q_arrays_list, axis=0)

        # 2. Mass 영역만 추출
        q_mass = q_avg_field[mass_dofs]

        # 3. 공간 통계 (Mean, Std, CoV)
        mean_val = np.mean(q_mass)
        std_val = np.std(q_mass)
        cov_val = std_val / mean_val if mean_val > 0 else float('inf')

        return mean_val, std_val, cov_val

    # A. Baseline: Standard 360 Rotation (모든 각도 1번씩)
    baseline_q_list = list(q_cache.values())
    base_mean, base_std, base_cov = calc_stats(baseline_q_list)

    # B. Target: Optimized Schedule (스케줄에 있는 각도들)
    # schedule = {0: "file_10.xdmf", 1: "file_30.xdmf", ...}
    target_q_list = []
    sorted_times = sorted(schedule.keys())

    for t in sorted_times:
        fname = schedule[t]
        # 파일명에서 각도 추출
        try:
            ang = int(fname.split('_')[-1].replace('.xdmf', ''))
        except:
            ang = 0

        if ang in q_cache:
            target_q_list.append(q_cache[ang])
        else:
            # 캐시에 없으면(그럴리 없겠지만) 0도 데이터라도 넣음
            if 0 in q_cache: target_q_list.append(q_cache[0])

    target_mean, target_std, target_cov = calc_stats(target_q_list)

    # -------------------------------------------------------------------------
    # 4. 결과 출력
    # -------------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "=" * 60)
        print("   SMART HEATING OPTIMIZATION REPORT")
        print("=" * 60)
        print(f"Metric: CoV (Coefficient of Variation) = StdDev / Mean")
        print(f"(Lower is Better. 0.0 = Perfectly Uniform)")
        print("-" * 60)

        print(f"1. BASELINE (Standard 360 Rotation):")
        print(f"   - Description: Rotating continuously (0~350 deg averaged)")
        print(f"   - Mean Q     : {base_mean:.2f}")
        print(f"   - CoV        : \033[93m{base_cov:.4f}\033[0m")  # Yellow

        print("-" * 60)

        print(f"2. TARGET (Optimized Schedule):")
        print(f"   - Description: Smart rotation ({len(target_q_list)} steps)")
        print(f"   - Mean Q     : {target_mean:.2f}")
        print(f"   - CoV        : \033[96m{target_cov:.4f}\033[0m")  # Cyan

        print("-" * 60)

        # 개선율 계산
        if base_cov > 0:
            improvement = (base_cov - target_cov) / base_cov * 100
            color = "\033[92m" if improvement > 0 else "\033[91m"  # Green or Red
            print(f"   => IMPROVEMENT: {color}{improvement:.2f}% Better than Standard Rotation\033[0m")
        else:
            print("   => IMPROVEMENT: N/A")

        print("=" * 60 + "\n")

#float
def analyze_heating_uniformity3(comm, schedule, rotation_center):
    """
    [수정됨] 전역 변수 ref_file을 사용하여 기준 메쉬를 로드합니다.
    """
    import h5py

    # 전역 변수 ref_file 사용
    if comm.rank == 0:
        print(f"\n[Analysis] Setting up Reference Mesh from {ref_file}...")

    with io.XDMFFile(comm, ref_file, "r") as xdmf:
        domain_ref = xdmf.read_mesh(name="mesh")
        cell_tags_ref = xdmf.read_meshtags(domain_ref, name="PhysicalGroups")

    id_mass = 1925
    mass_cells = cell_tags_ref.find(id_mass)
    if len(mass_cells) == 0:
        if comm.rank == 0: print("[Analysis] Error: Mass domain not found!")
        return

    Q_space_ref = fem.functionspace(domain_ref, ("DG", 0))
    q_coords_ref = Q_space_ref.tabulate_dof_coordinates()[:, :3]
    mass_dofs = mass_cells
    num_cells = len(q_coords_ref)

    # -------------------------------------------------------------------------
    # 2. 데이터 로딩 및 매핑 (캐싱)
    # -------------------------------------------------------------------------
    base_dir = os.path.dirname(ref_file)
    base_name = os.path.basename(ref_file)
    prefix = base_name.rsplit('_', 1)[0]

    all_angle_files = {}
    for ang in range(0, 360, 10):
        fname = os.path.join(base_dir, f"{prefix}_{ang}.xdmf")
        if os.path.exists(fname):
            all_angle_files[ang] = fname

    if comm.rank == 0:
        print(f"[Analysis] Found {len(all_angle_files)} angle files for Baseline calculation.")

    q_cache = {}

    for ang, fname in all_angle_files.items():
        h5_file = fname.replace(".xdmf", ".h5")
        q_vals_rot = np.zeros(num_cells)
        try:
            with h5py.File(h5_file, "r") as f:
                if "Function/real_HeatSource/0" in f:
                    q_vals_rot = f["Function/real_HeatSource/0"][:].flatten()
        except:
            pass

        angle_rad = np.radians(ang)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy, cz = rotation_center

        x_sh = q_coords_ref[:, 0] - cx
        y_sh = q_coords_ref[:, 1] - cy
        z_sh = q_coords_ref[:, 2] - cz
        x_rot = x_sh * c - z_sh * s
        z_rot = x_sh * s + z_sh * c

        with io.XDMFFile(comm, fname, "r") as xdmf_rot:
            domain_rot = xdmf_rot.read_mesh(name="mesh")

        tree = geometry.bb_tree(domain_rot, domain_rot.topology.dim)
        target_pts = np.zeros_like(q_coords_ref)
        target_pts[:, 0] = x_rot + cx
        target_pts[:, 1] = y_sh + cy
        target_pts[:, 2] = z_rot + cz

        cell_candidates = geometry.compute_collisions_points(tree, target_pts)
        colliding_cells = geometry.compute_colliding_cells(domain_rot, cell_candidates, target_pts)

        mapped_q = np.zeros(num_cells)
        for i in mass_dofs:
            links = colliding_cells.links(i)
            if len(links) > 0:
                mapped_q[i] = q_vals_rot[links[0]]

        q_cache[ang] = mapped_q
        if comm.rank == 0:
            print(f"  - Loaded & Mapped Angle {ang} deg", end="\r")

    if comm.rank == 0:
        print(f"\n[Analysis] Data caching complete. Calculating statistics...")

    # -------------------------------------------------------------------------
    # 3. 통계 계산 (가중 평균 지원)
    # -------------------------------------------------------------------------
    def calc_stats(q_arrays_list, weights=None):
        if not q_arrays_list: return 0.0, 0.0, float('inf')

        # 가중 평균 Q 필드 생성
        if weights is None:
            q_avg_field = np.mean(q_arrays_list, axis=0)
        else:
            q_avg_field = np.average(q_arrays_list, axis=0, weights=weights)

        q_mass = q_avg_field[mass_dofs]
        mean_val = np.mean(q_mass)
        std_val = np.std(q_mass)
        cov_val = std_val / mean_val if mean_val > 0 else float('inf')
        return mean_val, std_val, cov_val

    # A. Baseline (Uniform weights)
    baseline_q_list = list(q_cache.values())
    base_mean, base_std, base_cov = calc_stats(baseline_q_list)

    # B. Target (Optimized Schedule)
    target_q_list = []
    target_weights = []

    if isinstance(schedule, dict):
        sorted_times = sorted(schedule.keys())
        for t in sorted_times:
            fname = schedule[t]
            try:
                ang = int(fname.split('_')[-1].replace('.xdmf', ''))
            except:
                ang = 0
            if ang in q_cache:
                target_q_list.append(q_cache[ang])
                target_weights.append(1.0)
    elif isinstance(schedule, list):
        for fname, duration in schedule:
            try:
                ang = int(fname.split('_')[-1].replace('.xdmf', ''))
            except:
                ang = 0
            if ang in q_cache:
                target_q_list.append(q_cache[ang])
                target_weights.append(duration)

    target_mean, target_std, target_cov = calc_stats(target_q_list, weights=target_weights)

    # -------------------------------------------------------------------------
    # 4. 결과 출력
    # -------------------------------------------------------------------------
    if comm.rank == 0:
        print("\n" + "=" * 60)
        print("   SMART HEATING OPTIMIZATION REPORT")
        print("=" * 60)
        print(f"Metric: CoV (Coefficient of Variation) = StdDev / Mean")
        print(f"(Lower is Better. 0.0 = Perfectly Uniform)")
        print("-" * 60)
        print(f"1. BASELINE (Standard 360 Rotation):")
        print(f"   - Mean Q     : {base_mean:.2f}")
        print(f"   - CoV        : \033[93m{base_cov:.4f}\033[0m")
        print("-" * 60)
        print(f"2. TARGET (Optimized Schedule):")
        print(f"   - Steps      : {len(target_q_list)}")
        print(f"   - Mean Q     : {target_mean:.2f}")
        print(f"   - CoV        : \033[96m{target_cov:.4f}\033[0m")
        print("-" * 60)

        if base_cov > 0:
            improvement = (base_cov - target_cov) / base_cov * 100
            color = "\033[92m" if improvement > 0 else "\033[91m"
            print(f"   => IMPROVEMENT: {color}{improvement:.2f}% Better than Standard Rotation\033[0m")
        else:
            print("   => IMPROVEMENT: N/A")
        print("=" * 60 + "\n")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

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


def find_optimal_schedule_greedy(comm, schedule, rotation_center, total_steps=35):
    """
    그리디 알고리즘을 사용하여 목표 시간(total_steps) 동안
    가열 균일도(CoV)를 최소화하는 최적의 각도 조합(스케줄)을 찾습니다.
    """
    import h5py

    # 1. 기준 메쉬(0도) 로드 및 Mass 영역 찾기
    if comm.rank == 0:
        print(f"\n[Optimization] Loading data for optimization...")

    with io.XDMFFile(comm, ref_file, "r") as xdmf:
        domain_ref = xdmf.read_mesh(name="mesh")
        cell_tags_ref = xdmf.read_meshtags(domain_ref, name="PhysicalGroups")

    id_mass = 1925
    mass_cells = cell_tags_ref.find(id_mass)

    # Q 공간 및 좌표
    Q_space_ref = fem.functionspace(domain_ref, ("DG", 0))
    q_coords_ref = Q_space_ref.tabulate_dof_coordinates()[:, :3]

    # Mass 영역의 DOF 인덱스 (DG0이므로 셀 인덱스와 대응)
    mass_dofs = mass_cells

    # -------------------------------------------------------------------------
    # 2. 모든 각도의 Q 데이터를 메모리에 로드 (q_stack)
    # -------------------------------------------------------------------------
    sorted_times = sorted(schedule.keys())
    num_angles = len(sorted_times)
    num_cells = len(q_coords_ref)

    # 전체 Q 데이터를 담을 행렬 [각도인덱스, 전체셀]
    q_stack = np.zeros((num_angles, num_cells))

    for idx, t in enumerate(sorted_times):
        current_file = schedule[t]

        # 각도 계산
        try:
            angle_deg = float(current_file.split('_')[-1].replace('.xdmf', ''))
        except:
            angle_deg = 0.0
        angle_rad = np.radians(angle_deg)

        # HDF5 읽기
        h5_file = current_file.replace(".xdmf", ".h5")
        q_vals = np.zeros(num_cells)
        try:
            with h5py.File(h5_file, "r") as f:
                if "Function/real_HeatSource/0" in f:
                    q_vals = f["Function/real_HeatSource/0"][:].flatten()
        except:
            pass

        # 좌표 회전 및 매핑 (Ref 좌표계로 변환)
        # (간단한 Nearest Neighbor 매핑 - DG0이므로 중심점 기준)
        # 1) Ref 좌표를 회전시킴
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy, cz = rotation_center

        x_sh = q_coords_ref[:, 0] - cx
        y_sh = q_coords_ref[:, 1] - cy
        z_sh = q_coords_ref[:, 2] - cz

        # Y축 회전
        x_rot = x_sh * c - z_sh * s
        z_rot = x_sh * s + z_sh * c

        # 회전된 위치(target)가 Rotated Mesh의 어디에 있는지 찾아야 함
        # 하지만 여기서는 "Rotated Mesh의 값"을 "Ref Mesh"로 가져오는 것이므로
        # Rotated Mesh의 셀 구조는 Ref Mesh와 동일(회전만 됨)하다고 가정하고
        # 좌표 변환을 통해 매핑합니다.

        # *주의*: 엄밀하게는 bb_tree 충돌 감지를 해야 하지만,
        # 성능을 위해 analyze 함수와 동일한 로직(충돌 감지)을 사용합니다.

        # (메쉬 로드는 비용이 크므로, 여기서는 analyze 함수 로직을 그대로 차용)
        with io.XDMFFile(comm, current_file, "r") as xdmf_rot:
            domain_rot = xdmf_rot.read_mesh(name="mesh")

        tree = geometry.bb_tree(domain_rot, domain_rot.topology.dim)

        target_pts = np.zeros_like(q_coords_ref)
        target_pts[:, 0] = x_rot + cx
        target_pts[:, 1] = y_sh + cy
        target_pts[:, 2] = z_rot + cz

        cell_candidates = geometry.compute_collisions_points(tree, target_pts)
        colliding_cells = geometry.compute_colliding_cells(domain_rot, cell_candidates, target_pts)

        mapped_q = np.zeros(num_cells)
        # Mass 영역만 매핑해도 충분함
        for i in mass_dofs:
            links = colliding_cells.links(i)
            if len(links) > 0:
                mapped_q[i] = q_vals[links[0]]

        q_stack[idx, :] = mapped_q

        if comm.rank == 0:
            print(f"  - Loaded Angle {angle_deg} ({idx + 1}/{num_angles})", end="\r")

    # -------------------------------------------------------------------------
    # 3. 그리디 알고리즘 실행 (Mass 영역만 고려)
    # -------------------------------------------------------------------------
    q_stack_mass = q_stack[:, mass_dofs]  # [36, Mass셀개수]

    # 현재까지 누적된 Q 분포
    current_sum_Q = np.zeros(q_stack_mass.shape[1])

    selected_indices = []  # 선택된 각도 인덱스 리스트

    if comm.rank == 0:
        print(f"\n\n[Optimization] Finding best combination for {total_steps} steps...")

    for step in range(total_steps):
        best_idx = -1
        best_cov = float('inf')

        # 36개 각도를 각각 더해보고 CoV가 가장 낮아지는 것 선택
        for i in range(num_angles):
            trial_sum = current_sum_Q + q_stack_mass[i]

            # MPI 통신을 고려한 평균/표준편차 계산
            local_sum = np.sum(trial_sum)
            local_sq_sum = np.sum(trial_sum ** 2)
            local_count = len(trial_sum)

            global_sum = comm.allreduce(local_sum, op=MPI.SUM)
            global_sq_sum = comm.allreduce(local_sq_sum, op=MPI.SUM)
            global_count = comm.allreduce(local_count, op=MPI.SUM)

            mean_val = global_sum / global_count
            if mean_val > 0:
                var_val = (global_sq_sum / global_count) - (mean_val ** 2)
                std_val = np.sqrt(max(0, var_val))
                cov = std_val / mean_val
            else:
                cov = float('inf')

            if cov < best_cov:
                best_cov = cov
                best_idx = i

        # 선택 확정
        selected_indices.append(best_idx)
        current_sum_Q += q_stack_mass[best_idx]

        if comm.rank == 0:
            angle_key = sorted_times[best_idx]
            # 파일명에서 각도 추출
            fname = schedule[angle_key]
            ang = fname.split('_')[-1].replace('.xdmf', '')
            print(f"  Step {step + 1:02d}: Added Angle {ang:>3} deg | Resulting CoV: {best_cov:.4f}")

    # -------------------------------------------------------------------------
    # 4. 최적화된 스케줄 생성
    # -------------------------------------------------------------------------
    optimized_schedule = {}
    for i, idx in enumerate(selected_indices):
        # 시간 t = i (0, 1, 2, ...)
        # 파일 = 선택된 인덱스에 해당하는 파일
        original_key = sorted_times[idx]
        optimized_schedule[float(i)] = schedule[original_key]

    return optimized_schedule


def find_optimal_schedule_greedy_physics(comm, schedule, rotation_center, total_steps=35, dt=1.0, initial_temp=25.0):
    """
    [물리 기반 그리디 최적화]
    단순 Q 합산이 아니라, 매 스텝 실제 열전도 해석(FEM)을 수행하여
    '다음 스텝의 온도 분포(T)'가 가장 균일해지는 각도를 선택합니다.
    열전도(Diffusion) 효과를 고려하므로 핫스팟이 자연스럽게 퍼지는 현상까지 이용합니다.
    """
    import h5py
    from petsc4py import PETSc

    # -------------------------------------------------------------------------
    # 1. 초기 설정 (메쉬, 물성치, FEM 준비)
    # -------------------------------------------------------------------------
    if comm.rank == 0:
        print(f"\n[Optimization] Setting up Physics-Informed Optimization...")

    with io.XDMFFile(comm, ref_file, "r") as xdmf:
        domain_ref = xdmf.read_mesh(name="mesh")
        cell_tags_ref = xdmf.read_meshtags(domain_ref, name="PhysicalGroups")

    # ID 정의
    id_glass = 1924
    id_mass = 1925
    id_air = 1930

    mass_cells = cell_tags_ref.find(id_mass)
    if len(mass_cells) == 0:
        if comm.rank == 0: print("Error: Mass cells not found.")
        return {}

    # Function Space
    V = fem.functionspace(domain_ref, ("Lagrange", 1))  # 온도 T
    Q_space = fem.functionspace(domain_ref, ("DG", 0))  # 열원 Q

    # 물성치 설정 (기본값 Air)
    D0 = fem.functionspace(domain_ref, ("DG", 0))
    rho = fem.Function(D0);
    rho.x.array[:] = 1.2
    cp = fem.Function(D0);
    cp.x.array[:] = 1000.0
    k_therm = fem.Function(D0);
    k_therm.x.array[:] = 0.026

    # 영역별 물성 할당
    def assign_prop(tag, r, c, k):
        cells = cell_tags_ref.find(tag)
        if len(cells) > 0:
            rho.x.array[cells] = r
            cp.x.array[cells] = c
            k_therm.x.array[cells] = k

    assign_prop(id_glass, 2500, 840, 1.0)
    assign_prop(id_mass, 1000, 3500, 0.5)

    # -------------------------------------------------------------------------
    # 2. 모든 각도의 Q 데이터 미리 로드 (메모리 캐싱)
    # -------------------------------------------------------------------------
    q_coords = Q_space.tabulate_dof_coordinates()[:, :3]
    num_cells = len(q_coords)
    sorted_times = sorted(schedule.keys())
    num_angles = len(sorted_times)

    # [각도인덱스, 전체셀]
    q_stack = np.zeros((num_angles, num_cells))

    # Mass 영역의 DOF 인덱스 (DG0이므로 셀 인덱스와 동일)
    mass_dofs = mass_cells

    for idx, t in enumerate(sorted_times):
        current_file = schedule[t]
        try:
            angle_deg = float(current_file.split('_')[-1].replace('.xdmf', ''))
        except:
            angle_deg = 0.0
        angle_rad = np.radians(angle_deg)

        # HDF5 읽기
        h5_file = current_file.replace(".xdmf", ".h5")
        q_vals = np.zeros(num_cells)
        try:
            with h5py.File(h5_file, "r") as f:
                if "Function/real_HeatSource/0" in f:
                    q_vals = f["Function/real_HeatSource/0"][:].flatten()
        except:
            pass

        # 회전 매핑 (Ref 좌표계로)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy, cz = rotation_center

        x_sh = q_coords[:, 0] - cx
        y_sh = q_coords[:, 1] - cy
        z_sh = q_coords[:, 2] - cz

        x_rot = x_sh * c - z_sh * s
        z_rot = x_sh * s + z_sh * c

        # 매핑용 임시 로드
        with io.XDMFFile(comm, current_file, "r") as xdmf_rot:
            domain_rot = xdmf_rot.read_mesh(name="mesh")

        tree = geometry.bb_tree(domain_rot, domain_rot.topology.dim)
        target_pts = np.zeros_like(q_coords)
        target_pts[:, 0] = x_rot + cx
        target_pts[:, 1] = y_sh + cy
        target_pts[:, 2] = z_rot + cz

        cell_candidates = geometry.compute_collisions_points(tree, target_pts)
        colliding_cells = geometry.compute_colliding_cells(domain_rot, cell_candidates, target_pts)

        mapped_q = np.zeros(num_cells)
        # Mass 영역만 매핑 (속도 최적화)
        for i in mass_dofs:
            links = colliding_cells.links(i)
            if len(links) > 0:
                mapped_q[i] = q_vals[links[0]]

        q_stack[idx, :] = mapped_q

        if comm.rank == 0:
            print(f"  - Pre-loading Physics Data: Angle {angle_deg} ({idx + 1}/{num_angles})", end="\r")

    # -------------------------------------------------------------------------
    # 3. FEM Solver 설정 (행렬 A 미리 조립)
    # -------------------------------------------------------------------------
    if comm.rank == 0:
        print(f"\n[Optimization] Assembling system matrix for fast solving...")

    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # 현재 온도 (업데이트 됨)
    T_n = fem.Function(V)
    T_n.x.array[:] = initial_temp

    # 후보 열원 (매번 바뀜)
    Q_cand = fem.Function(Q_space)

    # 열 방정식 (Implicit Euler)
    # (rho*cp/dt)*T - k*laplacian(T) = (rho*cp/dt)*T_n + Q
    # Bilinear form a (좌변, 상수)
    a = (rho * cp / dt) * ufl.inner(T, v) * ufl.dx + \
        k_therm * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx

    # Linear form L (우변, 변수)
    L = (rho * cp / dt) * ufl.inner(T_n, v) * ufl.dx + \
        ufl.inner(Q_cand, v) * ufl.dx

    petsc_options_prefix_heat2 = "pc_"
    # System Matrix A 조립 (한 번만 수행)
    problem = fem.petsc.LinearProblem(a, L, bcs=[],
                                      petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                     "pc_factor_mat_solver_type": "mumps"},
                                      petsc_options_prefix=petsc_options_prefix_heat2)

    # A 행렬 조립 및 Solver 캐싱을 위해 한 번 더미 실행
    # (dolfinx LinearProblem은 solve() 호출 시 A가 변하지 않았으면 재사용함)
    # 하지만 명시적으로 solver를 꺼내 쓰는게 가장 빠름. 여기서는 LinearProblem의 캐싱 기능 활용.

    # -------------------------------------------------------------------------
    # 4. 물리 기반 그리디 루프
    # -------------------------------------------------------------------------
    selected_indices = []

    # Mass 영역의 DOF 인덱스 (V 공간) - CoV 계산용
    # locate_dofs_topological 사용
    domain_ref.topology.create_connectivity(domain_ref.topology.dim, domain_ref.topology.dim)
    mass_dofs_V = fem.locate_dofs_topological(V, domain_ref.topology.dim, mass_cells)

    if comm.rank == 0:
        print(f"[Optimization] Starting Physics-Informed Greedy Search ({total_steps} steps)...")

    for step in range(total_steps):
        best_idx = -1
        best_cov = float('inf')
        best_T_array = None

        # 현재 T_n 상태 저장 (복구용)
        current_T_array = T_n.x.array.copy()

        # 36개 각도를 각각 시뮬레이션 해봄
        for i in range(num_angles):
            # 1) 해당 각도의 Q 적용
            Q_cand.x.array[:] = q_stack[i, :]

            # 2) 1스텝 해석 (T_n -> T_next)
            # T_n 값은 고정된 상태에서 solve 호출
            # LinearProblem은 L을 다시 조립하고, A는 재사용하여 풉니다.
            T_next = problem.solve()

            # 3) 결과의 CoV 계산 (Mass 영역만)
            T_vals = T_next.x.array[mass_dofs_V]

            # MPI 통신
            local_sum = np.sum(T_vals)
            local_sq_sum = np.sum(T_vals ** 2)
            local_cnt = len(T_vals)

            g_sum = comm.allreduce(local_sum, op=MPI.SUM)
            g_sq_sum = comm.allreduce(local_sq_sum, op=MPI.SUM)
            g_cnt = comm.allreduce(local_cnt, op=MPI.SUM)

            mean_val = g_sum / g_cnt
            if mean_val > 0:
                var_val = (g_sq_sum / g_cnt) - (mean_val ** 2)
                std_val = np.sqrt(max(0, var_val))
                cov = std_val / mean_val
            else:
                cov = float('inf')

            # 4) 최적값 갱신
            if cov < best_cov:
                best_cov = cov
                best_idx = i
                # 최적의 온도 분포 저장 (다음 스텝의 초기값으로 쓰기 위해)
                best_T_array = T_next.x.array.copy()

            # T_n 복구 불필요 (problem.solve는 새로운 벡터를 반환하거나 내부 u를 갱신함.
            # 하지만 L form에서 T_n을 참조하므로 T_n 객체 자체는 루프 내에서 변하면 안됨)

        # 루프 끝, 최적의 각도 선택됨
        selected_indices.append(best_idx)

        # T_n을 최적의 결과로 업데이트 (다음 스텝의 시작점)
        T_n.x.array[:] = best_T_array

        if comm.rank == 0:
            angle_key = sorted_times[best_idx]
            fname = schedule[angle_key]
            ang = fname.split('_')[-1].replace('.xdmf', '')
            print(f"  Step {step + 1:02d}: Angle {ang:>3} deg | Predicted Temp CoV: {best_cov:.5f}")

    # -------------------------------------------------------------------------
    # 5. 스케줄 생성
    # -------------------------------------------------------------------------
    optimized_schedule = {}
    for i, idx in enumerate(selected_indices):
        original_key = sorted_times[idx]
        optimized_schedule[float(i)] = schedule[original_key]

    return optimized_schedule

def find_optimal_schedule_scipy(comm, schedule, rotation_center, total_steps=35):
    """
    수학적 최적화(SLSQP)를 사용하여 가열 균일도를 극대화하는 각도 조합을 찾습니다.
    그리디 알고리즘보다 더 강력하며, 이론적 한계치에 가까운 결과를 도출합니다.
    """
    import h5py
    from scipy.optimize import minimize

    # 1. 기준 메쉬(0도) 로드 및 Mass 영역 찾기
    ref_file = schedule[0]
    if comm.rank == 0:
        print(f"\n[Optimization] Loading data for Scipy Optimization...")

    with io.XDMFFile(comm, ref_file, "r") as xdmf:
        domain_ref = xdmf.read_mesh(name="mesh")
        cell_tags_ref = xdmf.read_meshtags(domain_ref, name="PhysicalGroups")

    id_mass = 1925
    mass_cells = cell_tags_ref.find(id_mass)

    # Q 공간 및 좌표
    Q_space_ref = fem.functionspace(domain_ref, ("DG", 0))
    q_coords_ref = Q_space_ref.tabulate_dof_coordinates()[:, :3]

    # Mass 영역의 DOF 인덱스
    mass_dofs = mass_cells

    # -------------------------------------------------------------------------
    # 2. 모든 각도의 Q 데이터를 메모리에 로드 (q_stack)
    # -------------------------------------------------------------------------
    sorted_times = sorted(schedule.keys())
    num_angles = len(sorted_times)
    num_cells = len(q_coords_ref)

    # 전체 Q 데이터를 담을 행렬 [각도인덱스, 전체셀]
    q_stack = np.zeros((num_angles, num_cells))

    for idx, t in enumerate(sorted_times):
        current_file = schedule[t]
        try:
            angle_deg = float(current_file.split('_')[-1].replace('.xdmf', ''))
        except:
            angle_deg = 0.0
        angle_rad = np.radians(angle_deg)

        # HDF5 읽기
        h5_file = current_file.replace(".xdmf", ".h5")
        q_vals = np.zeros(num_cells)
        try:
            with h5py.File(h5_file, "r") as f:
                if "Function/real_HeatSource/0" in f:
                    q_vals = f["Function/real_HeatSource/0"][:].flatten()
        except:
            pass

        # 좌표 회전 및 매핑
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy, cz = rotation_center

        x_sh = q_coords_ref[:, 0] - cx
        y_sh = q_coords_ref[:, 1] - cy
        z_sh = q_coords_ref[:, 2] - cz

        x_rot = x_sh * c - z_sh * s
        z_rot = x_sh * s + z_sh * c

        # 매핑 (간이 방식)
        with io.XDMFFile(comm, current_file, "r") as xdmf_rot:
            domain_rot = xdmf_rot.read_mesh(name="mesh")

        tree = geometry.bb_tree(domain_rot, domain_rot.topology.dim)
        target_pts = np.zeros_like(q_coords_ref)
        target_pts[:, 0] = x_rot + cx
        target_pts[:, 1] = y_sh + cy
        target_pts[:, 2] = z_rot + cz

        cell_candidates = geometry.compute_collisions_points(tree, target_pts)
        colliding_cells = geometry.compute_colliding_cells(domain_rot, cell_candidates, target_pts)

        mapped_q = np.zeros(num_cells)
        for i in mass_dofs:
            links = colliding_cells.links(i)
            if len(links) > 0:
                mapped_q[i] = q_vals[links[0]]

        q_stack[idx, :] = mapped_q

        if comm.rank == 0:
            print(f"  - Loaded Angle {angle_deg} ({idx + 1}/{num_angles})", end="\r")

    # Mass 영역 데이터만 추출 [36, Mass셀개수]
    q_stack_mass = q_stack[:, mass_dofs]

    # -------------------------------------------------------------------------
    # 3. Scipy 최적화 실행
    # -------------------------------------------------------------------------
    if comm.rank == 0:
        print(f"\n\n[Optimization] Running mathematical optimization (SLSQP)...")

        # 목적 함수: 가중치 w가 주어졌을 때, 결합된 Q 필드의 CoV 계산
        def objective(weights):
            # weights: [w1, w2, ..., w36] (합은 1)
            # 결합된 Q 필드 = w1*Q1 + w2*Q2 + ...
            combined_Q = np.dot(weights, q_stack_mass)

            mean_val = np.mean(combined_Q)
            if mean_val <= 0: return 1e6
            std_val = np.std(combined_Q)
            return std_val / mean_val

        # 제약 조건: 가중치의 합 = 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        # 범위: 각 가중치는 0 ~ 1
        bounds = [(0.0, 1.0) for _ in range(num_angles)]

        # 초기값: 균등 분포 (일반 회전)
        init_guess = np.ones(num_angles) / num_angles

        # 최적화 수행
        result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 1000, 'ftol': 1e-6})

        best_weights = result.x
        print(f"  Optimization Success: {result.success}")
        print(f"  Theoretical Best CoV: {result.fun:.4f}")

        # -------------------------------------------------------------------------
        # 4. 가중치를 시간 스텝(초)으로 변환 (Discretization)
        # -------------------------------------------------------------------------
        # 예: 가중치 0.1이고 총 35초면 -> 3.5초 -> 반올림하여 4초 할당

        # 일단 단순 반올림
        raw_seconds = best_weights * total_steps
        int_seconds = np.round(raw_seconds).astype(int)

        # 총 시간 맞추기 (오차 보정)
        diff = total_steps - np.sum(int_seconds)
        if diff != 0:
            # 오차가 있으면 가장 가중치가 큰 곳에 더하거나 뺌
            idx_max = np.argmax(best_weights)
            int_seconds[idx_max] += diff

        # 스케줄 생성
        optimized_schedule = {}
        current_step = 0

        print("\n[Optimization] Generated Schedule:")
        for idx, secs in enumerate(int_seconds):
            if secs > 0:
                angle_key = sorted_times[idx]
                fname = schedule[angle_key]
                ang = fname.split('_')[-1].replace('.xdmf', '')
                print(f"  - Angle {ang:>3} deg: {secs} seconds")

                for _ in range(secs):
                    if current_step < total_steps:
                        optimized_schedule[float(current_step)] = fname
                        current_step += 1

        return optimized_schedule
    else:
        return {}


#can decimal
def find_optimal_schedule_scipy2(comm, schedule, rotation_center, total_steps=36):
    """
    수학적 최적화(SLSQP)를 사용하여 가열 균일도를 극대화하는 각도 조합을 찾습니다.
    [수정] 정수 반올림을 하지 않고, 소수점 시간(Variable Duration)을 그대로 반환합니다.
    반환 형식: [(file_path, duration_seconds), ...]
    """
    import h5py
    from scipy.optimize import minimize

    # 1. 기준 메쉬(0도) 로드 및 Mass 영역 찾기
    ref_file = schedule[0]
    if comm.rank == 0:
        print(f"\n[Optimization] Loading data for Scipy Optimization...")

    with io.XDMFFile(comm, ref_file, "r") as xdmf:
        domain_ref = xdmf.read_mesh(name="mesh")
        cell_tags_ref = xdmf.read_meshtags(domain_ref, name="PhysicalGroups")

    id_mass = 1925
    mass_cells = cell_tags_ref.find(id_mass)

    # Q 공간 및 좌표
    Q_space_ref = fem.functionspace(domain_ref, ("DG", 0))
    q_coords_ref = Q_space_ref.tabulate_dof_coordinates()[:, :3]
    mass_dofs = mass_cells

    # -------------------------------------------------------------------------
    # 2. 모든 각도의 Q 데이터를 메모리에 로드
    # -------------------------------------------------------------------------
    sorted_times = sorted(schedule.keys())
    num_angles = len(sorted_times)
    num_cells = len(q_coords_ref)

    q_stack = np.zeros((num_angles, num_cells))

    for idx, t in enumerate(sorted_times):
        current_file = schedule[t]
        try:
            angle_deg = float(current_file.split('_')[-1].replace('.xdmf', ''))
        except:
            angle_deg = 0.0
        angle_rad = np.radians(angle_deg)

        h5_file = current_file.replace(".xdmf", ".h5")
        q_vals = np.zeros(num_cells)
        try:
            with h5py.File(h5_file, "r") as f:
                if "Function/real_HeatSource/0" in f:
                    q_vals = f["Function/real_HeatSource/0"][:].flatten()
        except:
            pass

        # 좌표 회전 및 매핑 (간이 방식)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy, cz = rotation_center

        x_sh = q_coords_ref[:, 0] - cx
        y_sh = q_coords_ref[:, 1] - cy
        z_sh = q_coords_ref[:, 2] - cz

        x_rot = x_sh * c - z_sh * s
        z_rot = x_sh * s + z_sh * c

        with io.XDMFFile(comm, current_file, "r") as xdmf_rot:
            domain_rot = xdmf_rot.read_mesh(name="mesh")

        tree = geometry.bb_tree(domain_rot, domain_rot.topology.dim)
        target_pts = np.zeros_like(q_coords_ref)
        target_pts[:, 0] = x_rot + cx
        target_pts[:, 1] = y_sh + cy
        target_pts[:, 2] = z_rot + cz

        cell_candidates = geometry.compute_collisions_points(tree, target_pts)
        colliding_cells = geometry.compute_colliding_cells(domain_rot, cell_candidates, target_pts)

        mapped_q = np.zeros(num_cells)
        for i in mass_dofs:
            links = colliding_cells.links(i)
            if len(links) > 0:
                mapped_q[i] = q_vals[links[0]]

        q_stack[idx, :] = mapped_q

        if comm.rank == 0:
            print(f"  - Loaded Angle {angle_deg} ({idx + 1}/{num_angles})", end="\r")

    q_stack_mass = q_stack[:, mass_dofs]

    # -------------------------------------------------------------------------
    # 3. Scipy 최적화 실행
    # -------------------------------------------------------------------------
    if comm.rank == 0:
        print(f"\n\n[Optimization] Running mathematical optimization (SLSQP)...")

        def get_cov(weights):
            combined_Q = np.dot(weights, q_stack_mass)
            mean_val = np.mean(combined_Q)
            if mean_val <= 0: return 1e6
            std_val = np.std(combined_Q)
            return std_val / mean_val

        # 초기값: 균등 분포
        init_guess = np.ones(num_angles) / num_angles
        baseline_cov = get_cov(init_guess)
        print(f"  Baseline CoV (Uniform): {baseline_cov:.4f}")

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = [(0.0, 1.0) for _ in range(num_angles)]

        result = minimize(get_cov, init_guess, method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 1000, 'ftol': 1e-6})

        best_weights = result.x
        theoretical_cov = result.fun
        print(f"  Optimization Success: {result.success}")
        print(f"  Theoretical Best CoV: {theoretical_cov:.4f}")

        # -------------------------------------------------------------------------
        # 4. 소수점 시간 스케줄 생성 (Variable Duration)
        # -------------------------------------------------------------------------
        optimized_schedule = []

        print("\n[Optimization] Final Schedule (Variable Time):")
        for idx, weight in enumerate(best_weights):
            duration = weight * total_steps
            # 너무 짧은 시간(예: 0.001초 미만)은 무시하여 계산 효율성 확보
            if duration > 1e-3:
                angle_key = sorted_times[idx]
                fname = schedule[angle_key]
                ang = fname.split('_')[-1].replace('.xdmf', '')
                print(f"  - Angle {ang:>3} deg: {duration:.4f} seconds")
                optimized_schedule.append((fname, duration))

        return optimized_schedule
    else:
        return []


def solve_transient_heat_transfer_rotating3(
        comm,
        schedule,
        rotation_center=None,
        output_file="result3/temperature_evolution.xdmf",
        dt=1.0,
        total_time=10.0,
        initial_temp=25.0
):
    """
    회전하는 메쉬의 Q 데이터를 고정된 메쉬(t=0)로 매핑하여 열 해석을 수행합니다.
    (시각화용 변수를 분리하여 물리적 열 손실 없이 Air/Glass를 투명하게 만듭니다)
    """

    # 1. 기준 메쉬(Reference Mesh) 로드
    ref_key = 0 if 0 in schedule else 0.0
    ref_file = schedule[ref_key]

    print(f"[HeatSolver] Loading REFERENCE mesh from {ref_file}...")

    with io.XDMFFile(comm, ref_file, "r") as xdmf:
        domain_ref = xdmf.read_mesh(name="mesh")
        cell_tags_ref = xdmf.read_meshtags(domain_ref, name="PhysicalGroups")

    # -------------------------------------------------------------------------
    # [설정] ID 정의
    # -------------------------------------------------------------------------
    id_glass = 1924
    id_mass = 1925
    id_air = 1930
    ids_mass_list = [1925]

    # -------------------------------------------------------------------------
    # [추가] 회전 중심 자동 계산 로직
    # -------------------------------------------------------------------------
    if rotation_center is None:
        glass_cells = cell_tags_ref.find(id_glass)
        if len(glass_cells) > 0:
            domain_ref.topology.create_connectivity(domain_ref.topology.dim, 0)
            vertex_indices = mesh.entities_to_geometry(domain_ref, domain_ref.topology.dim, glass_cells, True)
            unique_vertices = np.unique(vertex_indices)
            coords = domain_ref.geometry.x[unique_vertices]
            min_pt = np.min(coords, axis=0)
            max_pt = np.max(coords, axis=0)
            rotation_center = (min_pt + max_pt) / 2.0
            rotation_center = rotation_center[:3]
            if comm.rank == 0:
                print(f"[HeatSolver] Auto-calculated Rotation Center (Glass): {rotation_center}")
        else:
            rotation_center = (0.155, 0.13, 0.155)
            if comm.rank == 0:
                print("[HeatSolver] Warning: Glass domain not found. Using default rotation center.")

    # 2. Function Space 정의
    V_ref = fem.functionspace(domain_ref, ("Lagrange", 1))
    Q_space_ref = fem.functionspace(domain_ref, ("DG", 0))

    # 3. 물성치 설정
    D0 = fem.functionspace(domain_ref, ("DG", 0))
    rho = fem.Function(D0);
    rho.x.array[:] = 1.2
    cp = fem.Function(D0);
    cp.x.array[:] = 1000.0
    k_therm = fem.Function(D0);
    k_therm.x.array[:] = 0.026

    def assign_prop(tag, r, c, k):
        cells = cell_tags_ref.find(tag)
        if len(cells) > 0:
            rho.x.array[cells] = r
            cp.x.array[cells] = c
            k_therm.x.array[cells] = k

    assign_prop(id_glass, 2500, 840, 1.0)
    assign_prop(id_mass, 1000, 3500, 0.5)

    # 초기 온도 (계산용)
    T_n = fem.Function(V_ref)
    T_n.name = "Temperature"
    T_n.x.array[:] = initial_temp

    # [수정] 시각화 저장용 변수 (계산에 영향 안 줌)
    T_vis = fem.Function(V_ref)
    T_vis.name = "Temperature"
    T_vis.x.array[:] = initial_temp

    # 4. 열 방정식 정의
    T = ufl.TrialFunction(V_ref)
    v = ufl.TestFunction(V_ref)
    Q_external = fem.Function(Q_space_ref)

    term_time = (rho * cp / dt) * ufl.inner(T - T_n, v) * ufl.dx
    term_diff = k_therm * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx
    term_source = ufl.inner(Q_external, v) * ufl.dx

    F = term_time + term_diff - term_source

    petsc_options_prefix_heat = "pc_heat"
    problem = fem.petsc.LinearProblem(ufl.lhs(F), ufl.rhs(F), bcs=[],
                                      petsc_options={"ksp_type": "cg", "pc_type": "gamg"},
                                      petsc_options_prefix=petsc_options_prefix_heat)

    # 5. 결과 저장용 파일
    xdmf_out = io.XDMFFile(comm, output_file, "w")
    xdmf_out.write_mesh(domain_ref)
    cell_tags_ref.name = "PhysicalGroups"
    xdmf_out.write_meshtags(cell_tags_ref, domain_ref.geometry)

    # 초기 상태 저장 (시각화용 변수 사용)
    xdmf_out.write_function(T_vis, 0.0)

    # 6. 시간 루프
    t = 0.0
    print(f"[HeatSolver] Starting simulation...")

    q_coords_ref = Q_space_ref.tabulate_dof_coordinates()[:, :3]

    # -------------------------------------------------------------------------
    # [시각화용] Mass가 아닌 영역의 노드(DOF) 찾기
    # -------------------------------------------------------------------------
    mass_cells_vis = np.array([], dtype=np.int32)
    for tag in ids_mass_list:
        found = cell_tags_ref.find(tag)
        if len(found) > 0:
            mass_cells_vis = np.concatenate((mass_cells_vis, found))

    # Connectivity 계산 (에러 방지)
    domain_ref.topology.create_connectivity(domain_ref.topology.dim, domain_ref.topology.dim)

    mass_dofs = fem.locate_dofs_topological(V_ref, domain_ref.topology.dim, mass_cells_vis)
    mass_dofs = np.unique(mass_dofs)

    all_dofs_indices = np.arange(len(T_n.x.array), dtype=np.int32)
    dofs_to_zero = np.setdiff1d(all_dofs_indices, mass_dofs)

    # 초기 시각화 변수도 0으로 정리
    if len(dofs_to_zero) > 0:
        T_vis.x.array[dofs_to_zero] = 0.0
        xdmf_out.write_function(T_vis, 0.0)  # 0초 다시 저장

    # -------------------------------------------------------------------------
    # [노이즈 제거용] Q 매핑 시 사용할 Mass 셀 인덱스
    # -------------------------------------------------------------------------
    all_indices_q = np.arange(len(Q_external.x.array), dtype=np.int32)
    non_mass_cells_q = np.setdiff1d(all_indices_q, mass_cells_vis)

    while t < total_time:
        t += dt

        # (1) 스케줄링
        times = sorted(schedule.keys())
        current_sched_time = times[0]
        for st in times:
            if st <= t:
                current_sched_time = st
            else:
                break
        current_file = schedule[current_sched_time]

        try:
            angle_deg = float(current_file.split('_')[-1].replace('.xdmf', ''))
        except:
            angle_deg = 0.0
        angle_rad = np.radians(angle_deg)

        print(f"[HeatSolver] t={t:.2f}s | Reading Q from '{current_file}' (Angle={angle_deg} deg)")

        # (2) HDF5 데이터 읽기
        with io.XDMFFile(comm, current_file, "r") as xdmf_rot:
            domain_rot = xdmf_rot.read_mesh(name="mesh")

        Q_space_rot = fem.functionspace(domain_rot, ("DG", 0))
        Q_rot = fem.Function(Q_space_rot)

        import h5py
        h5_file = current_file.replace(".xdmf", ".h5")
        try:
            with h5py.File(h5_file, "r") as f:
                if "Function/real_HeatSource/0" in f:
                    data_real = f["Function/real_HeatSource/0"][:]
                    data_imag = None
                    if "Function/imag_HeatSource/0" in f:
                        data_imag = f["Function/imag_HeatSource/0"][:]

                    flat_real = data_real.flatten()
                    if np.issubdtype(Q_rot.x.array.dtype, np.complexfloating):
                        if data_imag is not None:
                            Q_rot.x.array[:] = flat_real + 1j * data_imag.flatten()
                        else:
                            Q_rot.x.array[:] = flat_real + 0j
                    else:
                        Q_rot.x.array[:] = flat_real
                else:
                    Q_rot.x.array[:] = 0.0
        except Exception as e:
            print(f"[Error] HDF5 read failed: {e}")
            Q_rot.x.array[:] = 0.0

        # (3) 회전 매핑
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy, cz = rotation_center

        x_shifted = q_coords_ref[:, 0] - cx
        y_shifted = q_coords_ref[:, 1] - cy
        z_shifted = q_coords_ref[:, 2] - cz

        x_rot = x_shifted * c - z_shifted * s
        y_rot = y_shifted
        z_rot = x_shifted * s + z_shifted * c

        target_points = np.zeros_like(q_coords_ref)
        target_points[:, 0] = x_rot + cx
        target_points[:, 1] = y_rot + cy
        target_points[:, 2] = z_rot + cz

        # (4) 충돌 감지 및 값 할당
        tree = geometry.bb_tree(domain_rot, domain_rot.topology.dim)
        cell_candidates = geometry.compute_collisions_points(tree, target_points)
        colliding_cells = geometry.compute_colliding_cells(domain_rot, cell_candidates, target_points)

        Q_external.x.array[:] = 0.0
        found_cells = np.full(len(target_points), -1, dtype=np.int32)

        for i in range(len(target_points)):
            cells = colliding_cells.links(i)
            if len(cells) > 0:
                found_cells[i] = cells[0]

        valid_mask = found_cells != -1
        if np.any(valid_mask):
            source_values = Q_rot.x.array[found_cells[valid_mask]]
            Q_external.x.array[valid_mask] = source_values

        # [노이즈 제거] 음식(Mass) 외 영역 Q = 0
        if len(non_mass_cells_q) > 0:
            Q_external.x.array[non_mass_cells_q] = 0.0

        print(f"  Mapped Q values. Max Q in Mass: {np.max(Q_external.x.array):.2f}")

        # (5) 풀이
        problem.solve()

        # [중요] 계산 결과는 T_n에 저장 (다음 스텝 계산을 위해 물리적 값 유지)
        T_n.x.array[:] = problem.u.x.array[:]

        # ---------------------------------------------------------------------
        # [시각화 트릭] T_vis에 복사 후 Mass가 아닌 곳만 0으로 변경
        # ---------------------------------------------------------------------
        T_vis.x.array[:] = T_n.x.array[:]  # 복사
        if len(dofs_to_zero) > 0:
            T_vis.x.array[dofs_to_zero] = 0.0  # 시각화용 변수만 조작
        # ---------------------------------------------------------------------

        # (6) 결과 저장 (조작된 T_vis 저장)
        xdmf_out.write_function(T_vis, t)

        if comm.rank == 0:
            # 출력은 실제 Mass 온도(T_n) 기준
            print(f"  Step t={t:.2f}s | Max Temp (Mass) = {np.max(T_n.x.array):.2f} C")

    xdmf_out.close()
    print("[HeatSolver] Done.")

def solve_transient_heat_transfer_rotating4(
        comm,
        schedule,
        rotation_center=None,
        output_file="result3/temperature_evolution.xdmf",
        dt=1.0,
        total_time=10.0,
        initial_temp=25.0
):
    """
    회전하는 메쉬의 Q 데이터를 고정된 메쉬(t=0)로 매핑하여 열 해석을 수행합니다.
    * output_file: 원본 결과 (전체 도메인 온도)
    * output_file_noair: 시각화용 결과 (Air/Glass = 0)
    """

    # 1. 기준 메쉬(Reference Mesh) 로드
    ref_key = 0 if 0 in schedule else 0.0
    ref_file = schedule[ref_key]

    print(f"[HeatSolver] Loading REFERENCE mesh from {ref_file}...")

    with io.XDMFFile(comm, ref_file, "r") as xdmf:
        domain_ref = xdmf.read_mesh(name="mesh")
        cell_tags_ref = xdmf.read_meshtags(domain_ref, name="PhysicalGroups")

    # -------------------------------------------------------------------------
    # [설정] ID 정의
    # -------------------------------------------------------------------------
    id_glass = 1924
    id_mass = 1925
    id_air = 1930
    ids_mass_list = [1925]

    # -------------------------------------------------------------------------
    # [추가] 회전 중심 자동 계산 로직
    # -------------------------------------------------------------------------
    if rotation_center is None:
        glass_cells = cell_tags_ref.find(id_glass)
        if len(glass_cells) > 0:
            domain_ref.topology.create_connectivity(domain_ref.topology.dim, 0)
            vertex_indices = mesh.entities_to_geometry(domain_ref, domain_ref.topology.dim, glass_cells, True)
            unique_vertices = np.unique(vertex_indices)
            coords = domain_ref.geometry.x[unique_vertices]
            min_pt = np.min(coords, axis=0)
            max_pt = np.max(coords, axis=0)
            rotation_center = (min_pt + max_pt) / 2.0
            rotation_center = rotation_center[:3]
            if comm.rank == 0:
                print(f"[HeatSolver] Auto-calculated Rotation Center (Glass): {rotation_center}")
        else:
            rotation_center = (0.155, 0.13, 0.155)
            if comm.rank == 0:
                print("[HeatSolver] Warning: Glass domain not found. Using default rotation center.")

    # 2. Function Space 정의
    V_ref = fem.functionspace(domain_ref, ("Lagrange", 1))
    Q_space_ref = fem.functionspace(domain_ref, ("DG", 0))

    # 3. 물성치 설정
    D0 = fem.functionspace(domain_ref, ("DG", 0))
    rho = fem.Function(D0);
    rho.x.array[:] = 1.2
    cp = fem.Function(D0);
    cp.x.array[:] = 1000.0
    k_therm = fem.Function(D0);
    k_therm.x.array[:] = 0.026

    def assign_prop(tag, r, c, k):
        cells = cell_tags_ref.find(tag)
        if len(cells) > 0:
            rho.x.array[cells] = r
            cp.x.array[cells] = c
            k_therm.x.array[cells] = k

    assign_prop(id_glass, 2500, 840, 1.0)
    assign_prop(id_mass, 1000, 3500, 0.5)

    # 초기 온도 (계산용)
    T_n = fem.Function(V_ref)
    T_n.name = "Temperature"
    T_n.x.array[:] = initial_temp

    # [추가] 시각화 저장용 변수
    T_vis = fem.Function(V_ref)
    T_vis.name = "Temperature"
    T_vis.x.array[:] = initial_temp

    # 4. 열 방정식 정의
    T = ufl.TrialFunction(V_ref)
    v = ufl.TestFunction(V_ref)
    Q_external = fem.Function(Q_space_ref)

    term_time = (rho * cp / dt) * ufl.inner(T - T_n, v) * ufl.dx
    term_diff = k_therm * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx
    term_source = ufl.inner(Q_external, v) * ufl.dx

    F = term_time + term_diff - term_source

    petsc_options_prefix_heat = "pc_heat"
    problem = fem.petsc.LinearProblem(ufl.lhs(F), ufl.rhs(F), bcs=[],
                                      petsc_options={"ksp_type": "cg", "pc_type": "gamg"},
                                      petsc_options_prefix=petsc_options_prefix_heat)

    # -------------------------------------------------------------------------
    # 5. 결과 저장용 파일 (두 개의 파일 생성)
    # -------------------------------------------------------------------------
    # 파일명 생성: test.xdmf -> test_noair.xdmf
    base, ext = os.path.splitext(output_file)
    output_file_noair = f"{base}_noair{ext}"

    # (1) 원본 파일 (전체 데이터)
    xdmf_out = io.XDMFFile(comm, output_file, "w")
    xdmf_out.write_mesh(domain_ref)
    cell_tags_ref.name = "PhysicalGroups"
    xdmf_out.write_meshtags(cell_tags_ref, domain_ref.geometry)
    xdmf_out.write_function(T_n, 0.0)

    # (2) 시각화용 파일 (No Air)
    xdmf_out_noair = io.XDMFFile(comm, output_file_noair, "w")
    xdmf_out_noair.write_mesh(domain_ref)
    xdmf_out_noair.write_meshtags(cell_tags_ref, domain_ref.geometry)

    # 6. 시간 루프
    t = 0.0
    print(f"[HeatSolver] Starting simulation...")
    print(f"[HeatSolver] Outputs: '{output_file}' (Full), '{output_file_noair}' (Mass Only)")

    q_coords_ref = Q_space_ref.tabulate_dof_coordinates()[:, :3]

    # -------------------------------------------------------------------------
    # [시각화용] Mass가 아닌 영역의 노드(DOF) 찾기
    # -------------------------------------------------------------------------
    mass_cells_vis = np.array([], dtype=np.int32)
    for tag in ids_mass_list:
        found = cell_tags_ref.find(tag)
        if len(found) > 0:
            mass_cells_vis = np.concatenate((mass_cells_vis, found))

    # Connectivity 계산 (에러 방지)
    domain_ref.topology.create_connectivity(domain_ref.topology.dim, domain_ref.topology.dim)

    mass_dofs = fem.locate_dofs_topological(V_ref, domain_ref.topology.dim, mass_cells_vis)
    mass_dofs = np.unique(mass_dofs)

    all_dofs_indices = np.arange(len(T_n.x.array), dtype=np.int32)
    dofs_to_zero = np.setdiff1d(all_dofs_indices, mass_dofs)

    # 초기 시각화 변수도 0으로 정리 후 저장
    if len(dofs_to_zero) > 0:
        T_vis.x.array[dofs_to_zero] = 0.0
    xdmf_out_noair.write_function(T_vis, 0.0)

    # -------------------------------------------------------------------------
    # [노이즈 제거용] Q 매핑 시 사용할 Mass 셀 인덱스
    # -------------------------------------------------------------------------
    all_indices_q = np.arange(len(Q_external.x.array), dtype=np.int32)
    non_mass_cells_q = np.setdiff1d(all_indices_q, mass_cells_vis)

    while t < total_time:
        t += dt

        # (1) 스케줄링
        times = sorted(schedule.keys())
        current_sched_time = times[0]
        for st in times:
            if st <= t:
                current_sched_time = st
            else:
                break
        current_file = schedule[current_sched_time]

        try:
            angle_deg = float(current_file.split('_')[-1].replace('.xdmf', ''))
        except:
            angle_deg = 0.0
        angle_rad = np.radians(angle_deg)

        print(f"[HeatSolver] t={t:.2f}s | Reading Q from '{current_file}' (Angle={angle_deg} deg)")

        # (2) HDF5 데이터 읽기
        with io.XDMFFile(comm, current_file, "r") as xdmf_rot:
            domain_rot = xdmf_rot.read_mesh(name="mesh")

        Q_space_rot = fem.functionspace(domain_rot, ("DG", 0))
        Q_rot = fem.Function(Q_space_rot)

        import h5py
        h5_file = current_file.replace(".xdmf", ".h5")
        try:
            with h5py.File(h5_file, "r") as f:
                if "Function/real_HeatSource/0" in f:
                    data_real = f["Function/real_HeatSource/0"][:]
                    data_imag = None
                    if "Function/imag_HeatSource/0" in f:
                        data_imag = f["Function/imag_HeatSource/0"][:]

                    flat_real = data_real.flatten()
                    if np.issubdtype(Q_rot.x.array.dtype, np.complexfloating):
                        if data_imag is not None:
                            Q_rot.x.array[:] = flat_real + 1j * data_imag.flatten()
                        else:
                            Q_rot.x.array[:] = flat_real + 0j
                    else:
                        Q_rot.x.array[:] = flat_real
                else:
                    Q_rot.x.array[:] = 0.0
        except Exception as e:
            print(f"[Error] HDF5 read failed: {e}")
            Q_rot.x.array[:] = 0.0

        # (3) 회전 매핑
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy, cz = rotation_center

        x_shifted = q_coords_ref[:, 0] - cx
        y_shifted = q_coords_ref[:, 1] - cy
        z_shifted = q_coords_ref[:, 2] - cz

        x_rot = x_shifted * c - z_shifted * s
        y_rot = y_shifted
        z_rot = x_shifted * s + z_shifted * c

        target_points = np.zeros_like(q_coords_ref)
        target_points[:, 0] = x_rot + cx
        target_points[:, 1] = y_rot + cy
        target_points[:, 2] = z_rot + cz

        # (4) 충돌 감지 및 값 할당
        tree = geometry.bb_tree(domain_rot, domain_rot.topology.dim)
        cell_candidates = geometry.compute_collisions_points(tree, target_points)
        colliding_cells = geometry.compute_colliding_cells(domain_rot, cell_candidates, target_points)

        Q_external.x.array[:] = 0.0
        found_cells = np.full(len(target_points), -1, dtype=np.int32)

        for i in range(len(target_points)):
            cells = colliding_cells.links(i)
            if len(cells) > 0:
                found_cells[i] = cells[0]

        valid_mask = found_cells != -1
        if np.any(valid_mask):
            source_values = Q_rot.x.array[found_cells[valid_mask]]
            Q_external.x.array[valid_mask] = source_values

        # [노이즈 제거] 음식(Mass) 외 영역 Q = 0
        if len(non_mass_cells_q) > 0:
            Q_external.x.array[non_mass_cells_q] = 0.0

        print(f"  Mapped Q values. Max Q in Mass: {np.max(Q_external.x.array):.2f}")

        # (5) 풀이
        problem.solve()

        # [중요] 계산 결과는 T_n에 저장 (다음 스텝 계산을 위해 물리적 값 유지)
        T_n.x.array[:] = problem.u.x.array[:]

        # (1) 원본 저장
        xdmf_out.write_function(T_n, t)

        # (2) 시각화용 저장 (복사 후 0으로 변경)
        T_vis.x.array[:] = T_n.x.array[:]  # 복사
        if len(dofs_to_zero) > 0:
            T_vis.x.array[dofs_to_zero] = 0.0  # 시각화용 변수만 조작
        xdmf_out_noair.write_function(T_vis, t)

        if comm.rank == 0:
            # 출력은 실제 Mass 온도(T_n) 기준
            print(f"  Step t={t:.2f}s | Max Temp (Mass) = {np.max(T_n.x.array):.2f} C")

    xdmf_out.close()
    xdmf_out_noair.close()
    print("[HeatSolver] Done.")


#float
def solve_transient_heat_transfer_rotating5(
        comm,
        schedule,
        rotation_center=None,
        output_file="result3/temperature_evolution.xdmf",
        dt=1.0,
        total_time=10.0,
        initial_temp=25.0
):
    """
    회전하는 메쉬의 Q 데이터를 고정된 메쉬(t=0)로 매핑하여 열 해석을 수행합니다.
    * schedule이 dict인 경우: 기존 방식 (고정 dt)
    * schedule이 list인 경우: [(file, duration)] 방식 (가변 dt)
    """
    # [수정] PETSc 임포트 (ScalarType 사용을 위해)
    from petsc4py import PETSc
    import numpy as np

    # 1. 기준 메쉬 로드
    # 스케줄 타입에 따라 첫 번째 파일 경로 추출
    if isinstance(schedule, dict):
        ref_key = 0 if 0 in schedule else min(schedule.keys())
        ref_file = schedule[ref_key]
    elif isinstance(schedule, list):
        ref_file = schedule[0][0]
    else:
        return

    print(f"[HeatSolver] Loading REFERENCE mesh from {ref_file}...")

    with io.XDMFFile(comm, ref_file, "r") as xdmf:
        domain_ref = xdmf.read_mesh(name="mesh")
        cell_tags_ref = xdmf.read_meshtags(domain_ref, name="PhysicalGroups")

    # -------------------------------------------------------------------------
    # [설정] ID 정의
    # -------------------------------------------------------------------------
    id_glass = 1924
    id_mass = 1925
    id_air = 1930
    ids_mass_list = [1925]

    if rotation_center is None:
        glass_cells = cell_tags_ref.find(id_glass)
        if len(glass_cells) > 0:
            domain_ref.topology.create_connectivity(domain_ref.topology.dim, 0)
            vertex_indices = mesh.entities_to_geometry(domain_ref, domain_ref.topology.dim, glass_cells, True)
            unique_vertices = np.unique(vertex_indices)
            coords = domain_ref.geometry.x[unique_vertices]
            min_pt = np.min(coords, axis=0)
            max_pt = np.max(coords, axis=0)
            rotation_center = (min_pt + max_pt) / 2.0
            rotation_center = rotation_center[:3]
            if comm.rank == 0:
                print(f"[HeatSolver] Auto-calculated Rotation Center (Glass): {rotation_center}")
        else:
            rotation_center = (0.155, 0.13, 0.155)
            if comm.rank == 0:
                print("[HeatSolver] Warning: Glass domain not found. Using default rotation center.")

    # 2. Function Space & 물성치
    V_ref = fem.functionspace(domain_ref, ("Lagrange", 1))
    Q_space_ref = fem.functionspace(domain_ref, ("DG", 0))
    D0 = fem.functionspace(domain_ref, ("DG", 0))

    rho = fem.Function(D0)
    rho.x.array[:] = 1.2

    cp = fem.Function(D0)
    cp.x.array[:] = 1000.0

    k_therm = fem.Function(D0)
    k_therm.x.array[:] = 0.026

    def assign_prop(tag, r, c, k):
        cells = cell_tags_ref.find(tag)
        if len(cells) > 0:
            rho.x.array[cells] = r
            cp.x.array[cells] = c
            k_therm.x.array[cells] = k

    assign_prop(id_glass, 2500, 840, 1.0)
    assign_prop(id_mass, 1000, 3500, 0.5)

    # 초기 온도
    T_n = fem.Function(V_ref)
    T_n.name = "Temperature"
    T_n.x.array[:] = initial_temp

    T_vis = fem.Function(V_ref)
    T_vis.name = "Temperature"
    T_vis.x.array[:] = initial_temp

    # 4. 열 방정식 정의
    T = ufl.TrialFunction(V_ref)
    v = ufl.TestFunction(V_ref)
    Q_external = fem.Function(Q_space_ref)

    # [수정] dt를 Constant로 정의할 때 PETSc.ScalarType으로 캐스팅하여 복소수 호환성 확보
    dt_const = fem.Constant(domain_ref, PETSc.ScalarType(dt))

    term_time = (rho * cp / dt_const) * ufl.inner(T - T_n, v) * ufl.dx
    term_diff = k_therm * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx
    term_source = ufl.inner(Q_external, v) * ufl.dx

    F = term_time + term_diff - term_source

    petsc_options_prefix_heat = "pc_heat"
    problem = fem.petsc.LinearProblem(ufl.lhs(F), ufl.rhs(F), bcs=[],
                                      petsc_options={"ksp_type": "cg", "pc_type": "gamg"},
                                      petsc_options_prefix=petsc_options_prefix_heat)

    # 5. 결과 저장용 파일
    base, ext = os.path.splitext(output_file)
    output_file_noair = f"{base}_noair{ext}"

    xdmf_out = io.XDMFFile(comm, output_file, "w")
    xdmf_out.write_mesh(domain_ref)
    cell_tags_ref.name = "PhysicalGroups"
    xdmf_out.write_meshtags(cell_tags_ref, domain_ref.geometry)
    xdmf_out.write_function(T_n, 0.0)

    xdmf_out_noair = io.XDMFFile(comm, output_file_noair, "w")
    xdmf_out_noair.write_mesh(domain_ref)
    xdmf_out_noair.write_meshtags(cell_tags_ref, domain_ref.geometry)

    # 6. 스케줄 리스트 변환
    if isinstance(schedule, dict):
        sorted_keys = sorted(schedule.keys())
        step_list = [(schedule[k], dt) for k in sorted_keys]
    else:
        step_list = schedule

    # -------------------------------------------------------------------------
    # 시각화 및 분석 준비
    # -------------------------------------------------------------------------
    q_coords_ref = Q_space_ref.tabulate_dof_coordinates()[:, :3]
    dx_mass = ufl.Measure("dx", domain=domain_ref, subdomain_data=cell_tags_ref, subdomain_id=id_mass)

    vol_form = fem.form(fem.Constant(domain_ref, PETSc.ScalarType(1.0)) * dx_mass)
    local_vol = fem.assemble_scalar(vol_form)
    total_mass_vol = comm.allreduce(local_vol, op=MPI.SUM)

    # [수정] 복소수 모드에서 Volume이 complex로 계산될 수 있으므로 실수부만 취함
    # 이 부분이 TypeError: '>' not supported between instances of 'complex' and 'int' 에러를 해결합니다.
    total_mass_vol = np.real(total_mass_vol)

    form_T_sum = fem.form(T_n * dx_mass)
    form_T_sq_sum = fem.form(T_n ** 2 * dx_mass)

    mass_cells_vis = np.array([], dtype=np.int32)
    for tag in ids_mass_list:
        found = cell_tags_ref.find(tag)
        if len(found) > 0: mass_cells_vis = np.concatenate((mass_cells_vis, found))

    domain_ref.topology.create_connectivity(domain_ref.topology.dim, domain_ref.topology.dim)
    mass_dofs = fem.locate_dofs_topological(V_ref, domain_ref.topology.dim, mass_cells_vis)
    mass_dofs = np.unique(mass_dofs)
    all_dofs_indices = np.arange(len(T_n.x.array), dtype=np.int32)
    dofs_to_zero = np.setdiff1d(all_dofs_indices, mass_dofs)

    if len(dofs_to_zero) > 0:
        T_vis.x.array[dofs_to_zero] = 0.0
    xdmf_out_noair.write_function(T_vis, 0.0)

    all_indices_q = np.arange(len(Q_external.x.array), dtype=np.int32)
    non_mass_cells_q = np.setdiff1d(all_indices_q, mass_cells_vis)

    # -------------------------------------------------------------------------
    # 시간 루프 시작
    # -------------------------------------------------------------------------
    t = 0.0
    print(f"[HeatSolver] Starting simulation with {len(step_list)} steps...")

    import h5py  # 루프 내 import

    for current_file, duration in step_list:
        # 현재 스텝의 dt 설정 (복소수 타입으로 업데이트)
        dt_step = float(duration)
        dt_const.value = PETSc.ScalarType(dt_step)
        t += dt_step

        try:
            angle_deg = float(current_file.split('_')[-1].replace('.xdmf', ''))
        except:
            angle_deg = 0.0
        angle_rad = np.radians(angle_deg)

        print(f"[HeatSolver] t={t:.2f}s (+{dt_step:.2f}s) | Reading Q from '{current_file}' (Angle={angle_deg} deg)")

        # (2) HDF5 데이터 읽기
        with io.XDMFFile(comm, current_file, "r") as xdmf_rot:
            domain_rot = xdmf_rot.read_mesh(name="mesh")

        Q_space_rot = fem.functionspace(domain_rot, ("DG", 0))
        Q_rot = fem.Function(Q_space_rot)

        h5_file = current_file.replace(".xdmf", ".h5")
        try:
            with h5py.File(h5_file, "r") as f:
                if "Function/real_HeatSource/0" in f:
                    data_real = f["Function/real_HeatSource/0"][:]
                    data_imag = None
                    if "Function/imag_HeatSource/0" in f:
                        data_imag = f["Function/imag_HeatSource/0"][:]

                    flat_real = data_real.flatten()
                    if np.issubdtype(Q_rot.x.array.dtype, np.complexfloating):
                        if data_imag is not None:
                            Q_rot.x.array[:] = flat_real + 1j * data_imag.flatten()
                        else:
                            Q_rot.x.array[:] = flat_real + 0j
                    else:
                        Q_rot.x.array[:] = flat_real
                else:
                    Q_rot.x.array[:] = 0.0
        except:
            Q_rot.x.array[:] = 0.0

        # (3) 회전 매핑
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy, cz = rotation_center
        x_shifted = q_coords_ref[:, 0] - cx
        y_shifted = q_coords_ref[:, 1] - cy
        z_shifted = q_coords_ref[:, 2] - cz
        x_rot = x_shifted * c - z_shifted * s
        z_rot = x_shifted * s + z_shifted * c

        target_points = np.zeros_like(q_coords_ref)
        target_points[:, 0] = x_rot + cx
        target_points[:, 1] = y_shifted + cy
        target_points[:, 2] = z_rot + cz

        tree = geometry.bb_tree(domain_rot, domain_rot.topology.dim)
        cell_candidates = geometry.compute_collisions_points(tree, target_points)
        colliding_cells = geometry.compute_colliding_cells(domain_rot, cell_candidates, target_points)

        Q_external.x.array[:] = 0.0
        found_cells = np.full(len(target_points), -1, dtype=np.int32)
        for i in range(len(target_points)):
            cells = colliding_cells.links(i)
            if len(cells) > 0: found_cells[i] = cells[0]

        valid_mask = found_cells != -1
        if np.any(valid_mask):
            Q_external.x.array[valid_mask] = Q_rot.x.array[found_cells[valid_mask]]

        if len(non_mass_cells_q) > 0:
            Q_external.x.array[non_mass_cells_q] = 0.0

        # (5) 풀이
        problem.solve()
        T_n.x.array[:] = problem.u.x.array[:]

        # [통계] CoV 계산
        local_T_sum = fem.assemble_scalar(form_T_sum)
        local_T_sq_sum = fem.assemble_scalar(form_T_sq_sum)
        global_T_sum = comm.allreduce(local_T_sum, op=MPI.SUM)
        global_T_sq_sum = comm.allreduce(local_T_sq_sum, op=MPI.SUM)

        if total_mass_vol > 0:
            T_mean = global_T_sum / total_mass_vol
            T_var = (global_T_sq_sum / total_mass_vol) - (T_mean ** 2)
            # 복소수 오차 방지를 위해 real 취함
            T_var = T_var.real if np.iscomplexobj(T_var) else T_var
            T_std = np.sqrt(max(0, T_var))
            T_mean_real = T_mean.real if np.iscomplexobj(T_mean) else T_mean
            T_cov = T_std / T_mean_real if T_mean_real > 0 else 0.0
        else:
            T_cov = 0.0

        # (6) 결과 저장
        xdmf_out.write_function(T_n, t)
        T_vis.x.array[:] = T_n.x.array[:]
        if len(dofs_to_zero) > 0: T_vis.x.array[dofs_to_zero] = 0.0
        xdmf_out_noair.write_function(T_vis, t)

        if comm.rank == 0:
            # 출력 시 복소수 경고 방지를 위해 real 값 사용
            max_temp = np.max(T_n.x.array).real
            print(f"  Step t={t:.2f}s | Max T: {max_temp:.1f} C | CoV: \033[96m{T_cov:.4f}\033[0m")

    xdmf_out.close()
    xdmf_out_noair.close()
    print("[HeatSolver] Done.")


def evaluate_temperature_uniformity(comm, xdmf_path, description="", target_time=36.0):
    """
    열 해석 결과 파일(.h5)을 직접 읽어, 지정된 시간(target_time)의 최종 온도 분포에 대한
    통계(최소/최대/평균/CoV)를 계산하고 출력합니다.
    [수정] XDMFFile.read_function 대신 h5py를 사용하여 데이터를 직접 로드합니다.
    """

    h5_path = xdmf_path.replace(".xdmf", ".h5")
    if not os.path.exists(h5_path):
        if comm.rank == 0:
            print(f"\n[Evaluation] \033[91mError: Result file not found: {h5_path}\033[0m")
        return

    if comm.rank == 0:
        print("\n" + "=" * 60)
        print(f"   FINAL TEMPERATURE UNIFORMITY REPORT")
        print(f"   Schedule: {description}")
        print("=" * 60)

    try:
        # 1. 메쉬 로드 (토폴로지 정보 필요)
        with io.XDMFFile(comm, xdmf_path, "r") as xdmf:
            domain = xdmf.read_mesh(name="mesh")
            cell_tags = xdmf.read_meshtags(domain, name="PhysicalGroups")

        # 2. FunctionSpace 생성
        V = fem.functionspace(domain, ("Lagrange", 1))

        # 3. HDF5 파일에서 데이터 읽기
        T_values = None

        with h5py.File(h5_path, "r") as f:
            # 데이터셋 경로 찾기 (보통 /Function/real_Temperature/...)
            if "Function/real_Temperature" in f:
                group = f["Function/real_Temperature"]

                # 저장된 모든 시간 키 가져오기
                keys = list(group.keys())

                # target_time과 가장 가까운 시간 키 찾기
                best_key = None
                min_diff = 1e9

                for k in keys:
                    try:
                        # 키 포맷이 "0", "0_5", "10" 등으로 다양할 수 있음
                        # "."이 "_"로 바뀌어 저장되는 경우가 많음
                        t_val = float(k.replace('_', '.'))
                        diff = abs(t_val - target_time)
                        if diff < min_diff:
                            min_diff = diff
                            best_key = k
                    except:
                        continue

                if best_key:
                    if comm.rank == 0:
                        # print(f"[Evaluation] Reading time step key: '{best_key}' (Target: {target_time}s)")
                        pass

                    # 실수부 읽기
                    data_real = group[best_key][:]

                    # 허수부 읽기 (있다면)
                    data_imag = None
                    if "Function/imag_Temperature" in f:
                        imag_group = f["Function/imag_Temperature"]
                        if best_key in imag_group:
                            data_imag = imag_group[best_key][:]

                    # 데이터 합치기
                    flat_real = data_real.flatten()
                    if data_imag is not None:
                        T_values = flat_real + 1j * data_imag.flatten()
                    else:
                        T_values = flat_real
                else:
                    if comm.rank == 0:
                        print(
                            f"[Evaluation] Warning: No suitable time step found for {target_time}s. Keys: {keys[:5]}...")
            else:
                if comm.rank == 0:
                    print(f"[Evaluation] Error: 'real_Temperature' group not found in H5 file.")
                return

        if T_values is None:
            return

        # 4. Mass 영역의 DOF(노드) 찾기
        id_mass = 1925
        mass_cells = cell_tags.find(id_mass)
        if len(mass_cells) == 0:
            if comm.rank == 0: print("  Error: Mass domain not found in result file.")
            return

        domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)
        mass_dofs = fem.locate_dofs_topological(V, domain.topology.dim, mass_cells)

        # 5. Mass 영역의 온도 값만 추출
        # T_values는 전체 도메인의 값이므로 mass_dofs로 인덱싱
        T_values_mass = T_values[mass_dofs]

        # MPI 통신을 통해 전체 값 취합
        all_T_values = comm.allgather(T_values_mass)
        if comm.rank == 0:
            full_T_mass = np.concatenate(all_T_values)

            # 복소수일 경우 실수부만 취함
            if np.iscomplexobj(full_T_mass):
                full_T_mass = full_T_mass.real

            # 6. 통계 계산
            min_T = np.min(full_T_mass)
            max_T = np.max(full_T_mass)
            mean_T = np.mean(full_T_mass)
            std_T = np.std(full_T_mass)
            cov_T = std_T / mean_T if mean_T > 0 else float('inf')

            # 7. 결과 출력
            print(f"  - Min Temp    : {min_T:.2f} C")
            print(f"  - Max Temp    : {max_T:.2f} C")
            print(f"  - Mean Temp   : {mean_T:.2f} C")
            print(f"  - Std. Dev.   : {std_T:.2f}")
            print(f"  - CoV (T)     : \033[92m{cov_T:.4f}\033[0m  <-- Lower is Better")
            print("=" * 60 + "\n")

    except Exception as e:
        if comm.rank == 0:
            print(f"  \033[91mEvaluation failed: {e}\033[0m")
            print("=" * 60 + "\n")

def find_optimal_schedule_stochastic_physics(comm, schedule, rotation_center, total_steps=36, dt=1.0, initial_temp=25.0,
                                             num_trials=5, top_k=3):
    """
    [확률적 물리 기반 최적화]
    Greedy 방식의 한계(근시안적 선택)를 극복하기 위해,
    매 스텝 상위 k개의 각도 중 하나를 랜덤하게 선택하여 여러 번(num_trials) 시도합니다.
    그 중 최종 결과(Final CoV)가 가장 좋은 스케줄을 반환합니다.
    """


    # -------------------------------------------------------------------------
    # 1. 초기 설정 (메쉬, 물성치, FEM 준비) - 한 번만 수행
    # -------------------------------------------------------------------------
    if comm.rank == 0:
        print(f"\n[Optimization] Setting up Stochastic Physics Optimization ({num_trials} trials)...")

    # 전역 변수 ref_file 사용
    with io.XDMFFile(comm, ref_file, "r") as xdmf:
        domain_ref = xdmf.read_mesh(name="mesh")
        cell_tags_ref = xdmf.read_meshtags(domain_ref, name="PhysicalGroups")

    id_glass = 1924
    id_mass = 1925
    mass_cells = cell_tags_ref.find(id_mass)

    if len(mass_cells) == 0: return {}

    # Function Space
    V = fem.functionspace(domain_ref, ("Lagrange", 1))
    Q_space = fem.functionspace(domain_ref, ("DG", 0))
    D0 = fem.functionspace(domain_ref, ("DG", 0))

    # 물성치
    rho = fem.Function(D0);
    rho.x.array[:] = 1.2
    cp = fem.Function(D0);
    cp.x.array[:] = 1000.0
    k_therm = fem.Function(D0);
    k_therm.x.array[:] = 0.026

    def assign_prop(tag, r, c, k):
        cells = cell_tags_ref.find(tag)
        if len(cells) > 0:
            rho.x.array[cells] = r;
            cp.x.array[cells] = c;
            k_therm.x.array[cells] = k

    assign_prop(id_glass, 2500, 840, 1.0)
    assign_prop(id_mass, 1000, 3500, 0.5)

    # -------------------------------------------------------------------------
    # 2. Q 데이터 캐싱
    # -------------------------------------------------------------------------
    q_coords = Q_space.tabulate_dof_coordinates()[:, :3]
    num_cells = len(q_coords)
    sorted_times = sorted(schedule.keys())
    num_angles = len(sorted_times)
    q_stack = np.zeros((num_angles, num_cells))
    mass_dofs = mass_cells

    for idx, t in enumerate(sorted_times):
        current_file = schedule[t]
        try:
            angle_deg = float(current_file.split('_')[-1].replace('.xdmf', ''))
        except:
            angle_deg = 0.0
        angle_rad = np.radians(angle_deg)

        h5_file = current_file.replace(".xdmf", ".h5")
        q_vals = np.zeros(num_cells)
        try:
            with h5py.File(h5_file, "r") as f:
                if "Function/real_HeatSource/0" in f:
                    q_vals = f["Function/real_HeatSource/0"][:].flatten()
        except:
            pass

        c, s = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy, cz = rotation_center
        x_sh = q_coords[:, 0] - cx
        y_sh = q_coords[:, 1] - cy
        z_sh = q_coords[:, 2] - cz
        x_rot = x_sh * c - z_sh * s
        z_rot = x_sh * s + z_sh * c

        with io.XDMFFile(comm, current_file, "r") as xdmf_rot:
            domain_rot = xdmf_rot.read_mesh(name="mesh")
        tree = geometry.bb_tree(domain_rot, domain_rot.topology.dim)
        target_pts = np.zeros_like(q_coords)
        target_pts[:, 0] = x_rot + cx
        target_pts[:, 1] = y_sh + cy
        target_pts[:, 2] = z_rot + cz

        cell_candidates = geometry.compute_collisions_points(tree, target_pts)
        colliding_cells = geometry.compute_colliding_cells(domain_rot, cell_candidates, target_pts)

        mapped_q = np.zeros(num_cells)
        for i in mass_dofs:
            links = colliding_cells.links(i)
            if len(links) > 0: mapped_q[i] = q_vals[links[0]]
        q_stack[idx, :] = mapped_q
        if comm.rank == 0: print(f"  - Pre-loading Data: {angle_deg} deg ({idx + 1}/{num_angles})", end="\r")

    # -------------------------------------------------------------------------
    # 3. FEM Solver 준비
    # -------------------------------------------------------------------------
    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    T_n = fem.Function(V)
    Q_cand = fem.Function(Q_space)

    a = (rho * cp / dt) * ufl.inner(T, v) * ufl.dx + k_therm * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx
    L = (rho * cp / dt) * ufl.inner(T_n, v) * ufl.dx + ufl.inner(Q_cand, v) * ufl.dx

    problem = fem.petsc.LinearProblem(a, L, bcs=[],
                                      petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                     "pc_factor_mat_solver_type": "mumps"},
                                      petsc_options_prefix="pc_stoch")

    domain_ref.topology.create_connectivity(domain_ref.topology.dim, domain_ref.topology.dim)
    mass_dofs_V = fem.locate_dofs_topological(V, domain_ref.topology.dim, mass_cells)

    # -------------------------------------------------------------------------
    # 4. 확률적 탐색 루프 (Trials)
    # -------------------------------------------------------------------------
    global_best_cov = float('inf')
    global_best_schedule_indices = []

    for trial in range(num_trials):
        if comm.rank == 0: print(f"\n  [Trial {trial + 1}/{num_trials}] Simulating...")

        # 초기화
        T_n.x.array[:] = initial_temp
        current_indices = []

        for step in range(total_steps):
            candidates = []  # (cov, angle_index, T_array)

            # 모든 각도 시뮬레이션
            for i in range(num_angles):
                Q_cand.x.array[:] = q_stack[i, :]
                T_next = problem.solve()

                # CoV 계산
                T_vals = T_next.x.array[mass_dofs_V]
                # 복소수 처리
                if np.iscomplexobj(T_vals): T_vals = T_vals.real

                local_sum = np.sum(T_vals)
                local_sq_sum = np.sum(T_vals ** 2)
                local_cnt = len(T_vals)

                g_sum = comm.allreduce(local_sum, op=MPI.SUM)
                g_sq_sum = comm.allreduce(local_sq_sum, op=MPI.SUM)
                g_cnt = comm.allreduce(local_cnt, op=MPI.SUM)

                mean_val = g_sum / g_cnt
                if mean_val > 0:
                    var_val = (g_sq_sum / g_cnt) - (mean_val ** 2)
                    std_val = np.sqrt(max(0, var_val))
                    cov = std_val / mean_val
                else:
                    cov = 1e6

                # 후보 리스트에 저장 (메모리 절약을 위해 T_array는 꼭 필요할 때만 복사)
                # 여기서는 선택된 것만 복사하기 위해 일단 인덱스만 저장
                candidates.append((cov, i))

            # CoV 기준으로 정렬 (오름차순)
            candidates.sort(key=lambda x: x[0])

            # 상위 K개 중 랜덤 선택 (Stochastic)
            # 마지막 스텝에서는 무조건 1등을 뽑는게 유리할 수 있음
            if step == total_steps - 1:
                chosen = candidates[0]
            else:
                # top_k 범위 내에서 랜덤 선택
                limit = min(top_k, len(candidates))
                chosen = random.choice(candidates[:limit])

            chosen_cov, chosen_idx = chosen
            current_indices.append(chosen_idx)

            # 선택된 각도로 T_n 업데이트 (다음 스텝 준비)
            Q_cand.x.array[:] = q_stack[chosen_idx, :]
            T_final_step = problem.solve()
            T_n.x.array[:] = T_final_step.x.array[:]

            if comm.rank == 0 and step % 5 == 0:
                print(f"    Step {step + 1}: Selected Angle {sorted_times[chosen_idx]} (CoV: {chosen_cov:.4f})")

        # Trial 종료 후 최종 CoV 확인
        # 마지막 스텝의 CoV가 최종 성능 지표
        final_cov = chosen_cov  # 루프 마지막 값

        if comm.rank == 0:
            print(f"  => Trial {trial + 1} Final CoV: {final_cov:.5f}")

        if final_cov < global_best_cov:
            global_best_cov = final_cov
            global_best_schedule_indices = current_indices
            if comm.rank == 0: print(f"     (New Best Found!)")

    # -------------------------------------------------------------------------
    # 5. 최종 스케줄 생성
    # -------------------------------------------------------------------------
    optimized_schedule = {}
    for i, idx in enumerate(global_best_schedule_indices):
        original_key = sorted_times[idx]
        optimized_schedule[float(i)] = schedule[original_key]

    return optimized_schedule


def find_optimal_schedule_physics_informed_scipy(comm, schedule, rotation_center, total_steps=36, initial_temp=25.0):
    """
    [물리 기반 Scipy 최적화 - 최종 버전]
    Scipy의 목적 함수(objective) 자체를 '전체 물리 시뮬레이션'으로 정의합니다.
    Scipy가 제안하는 가중치(시간 배분)에 따라 36초 시뮬레이션을 끝까지 돌리고,
    그 결과로 나온 '최종 온도 분포의 CoV'를 점수로 반환합니다.
    계산 비용이 매우 높지만, 이론적으로 가장 정확한 최적해를 찾을 수 있습니다.
    """
    import h5py
    from scipy.optimize import minimize
    from petsc4py import PETSc
    import random

    # -------------------------------------------------------------------------
    # 1. 초기 설정 (메쉬, 물성치, Q 데이터 캐싱) - 한 번만 수행
    # -------------------------------------------------------------------------
    if comm.rank == 0:
        print(f"\n[Optimization] Setting up Physics-Informed Scipy (This will take a long time)...")

    with io.XDMFFile(comm, ref_file, "r") as xdmf:
        domain_ref = xdmf.read_mesh(name="mesh")
        cell_tags_ref = xdmf.read_meshtags(domain_ref, name="PhysicalGroups")

    id_mass = 1925
    mass_cells = cell_tags_ref.find(id_mass)
    if len(mass_cells) == 0: return []

    V = fem.functionspace(domain_ref, ("Lagrange", 1))
    Q_space = fem.functionspace(domain_ref, ("DG", 0))
    D0 = fem.functionspace(domain_ref, ("DG", 0))

    rho = fem.Function(D0);
    rho.x.array[:] = 1.2
    cp = fem.Function(D0);
    cp.x.array[:] = 1000.0
    k_therm = fem.Function(D0);
    k_therm.x.array[:] = 0.026

    def assign_prop(tag, r, c, k):
        cells = cell_tags_ref.find(tag)
        if len(cells) > 0:
            rho.x.array[cells] = r;
            cp.x.array[cells] = c;
            k_therm.x.array[cells] = k

    assign_prop(1924, 2500, 840, 1.0)
    assign_prop(id_mass, 1000, 3500, 0.5)

    q_coords = Q_space.tabulate_dof_coordinates()[:, :3]
    num_cells = len(q_coords)
    sorted_times = sorted(schedule.keys())
    num_angles = len(sorted_times)
    q_stack = np.zeros((num_angles, num_cells))

    for idx, t in enumerate(sorted_times):
        # (이전과 동일한 Q 데이터 로딩 및 매핑 로직)
        current_file = schedule[t]
        try:
            angle_deg = float(current_file.split('_')[-1].replace('.xdmf', ''))
        except:
            angle_deg = 0.0
        angle_rad = np.radians(angle_deg)
        h5_file = current_file.replace(".xdmf", ".h5")
        q_vals = np.zeros(num_cells)
        try:
            with h5py.File(h5_file, "r") as f:
                if "Function/real_HeatSource/0" in f: q_vals = f["Function/real_HeatSource/0"][:].flatten()
        except:
            pass
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy, cz = rotation_center
        x_sh = q_coords[:, 0] - cx;
        y_sh = q_coords[:, 1] - cy;
        z_sh = q_coords[:, 2] - cz
        x_rot = x_sh * c - z_sh * s;
        z_rot = x_sh * s + z_sh * c
        with io.XDMFFile(comm, current_file, "r") as xdmf_rot:
            domain_rot = xdmf_rot.read_mesh(name="mesh")
        tree = geometry.bb_tree(domain_rot, domain_rot.topology.dim)
        target_pts = np.zeros_like(q_coords);
        target_pts[:, 0] = x_rot + cx;
        target_pts[:, 1] = y_sh + cy;
        target_pts[:, 2] = z_rot + cz
        cell_candidates = geometry.compute_collisions_points(tree, target_pts)
        colliding_cells = geometry.compute_colliding_cells(domain_rot, cell_candidates, target_pts)
        mapped_q = np.zeros(num_cells)
        for i in mass_cells:
            links = colliding_cells.links(i)
            if len(links) > 0: mapped_q[i] = q_vals[links[0]]
        q_stack[idx, :] = mapped_q
        if comm.rank == 0: print(f"  - Pre-loading Data: {angle_deg} deg ({idx + 1}/{num_angles})", end="\r")

    # -------------------------------------------------------------------------
    # 2. Scipy 목적 함수 정의 (핵심)
    # -------------------------------------------------------------------------
    # 이 함수는 Rank 0에서만 호출되지만, 내부는 MPI 통신을 포함해야 합니다.

    # FEM Solver 준비
    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    T_n = fem.Function(V)
    Q_cand = fem.Function(Q_space)
    dt_const = fem.Constant(domain_ref, PETSc.ScalarType(1.0))
    a = (rho * cp / dt_const) * ufl.inner(T, v) * ufl.dx + k_therm * ufl.inner(ufl.grad(T), ufl.grad(v)) * ufl.dx
    L = (rho * cp / dt_const) * ufl.inner(T_n, v) * ufl.dx + ufl.inner(Q_cand, v) * ufl.dx
    problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                                                   "pc_factor_mat_solver_type": "mumps"},petsc_options_prefix="pc")
    domain_ref.topology.create_connectivity(domain_ref.topology.dim, domain_ref.topology.dim)
    mass_dofs_V = fem.locate_dofs_topological(V, domain_ref.topology.dim, mass_cells)

    # 목적 함수
    def objective_physics(weights):
        # 1. 가중치를 시간 스케줄로 변환
        temp_schedule = []
        for idx, weight in enumerate(weights):
            duration = weight * total_steps
            if duration > 1e-4:  # 매우 짧은 시간은 무시
                temp_schedule.append((schedule[sorted_times[idx]], duration))

        # 2. 인메모리(in-memory) 물리 시뮬레이션 수행
        T_n.x.array[:] = initial_temp
        for file_path, duration in temp_schedule:
            # dt 업데이트
            dt_const.value = PETSc.ScalarType(duration)

            # Q 업데이트
            try:
                ang_idx = sorted_times.index(int(file_path.split('_')[-2]))
            except:
                ang_idx = 0  # Fallback
            Q_cand.x.array[:] = q_stack[ang_idx, :]

            # 1 스텝 풀이
            T_next = problem.solve()
            T_n.x.array[:] = T_next.x.array[:]

        # 3. 최종 온도 분포의 CoV 계산
        T_vals = T_n.x.array[mass_dofs_V]
        if np.iscomplexobj(T_vals): T_vals = T_vals.real

        # MPI 통신으로 전역 CoV 계산
        local_sum = np.sum(T_vals);
        local_sq_sum = np.sum(T_vals ** 2);
        local_cnt = len(T_vals)
        g_sum = comm.allreduce(local_sum, op=MPI.SUM)
        g_sq_sum = comm.allreduce(local_sq_sum, op=MPI.SUM)
        g_cnt = comm.allreduce(local_cnt, op=MPI.SUM)

        mean_val = g_sum / g_cnt
        if mean_val > 0:
            var_val = (g_sq_sum / g_cnt) - (mean_val ** 2)
            std_val = np.sqrt(max(0, var_val))
            cov = std_val / mean_val
        else:
            cov = 1e6

        if comm.rank == 0:
            print(f"    Scipy Trial -> Final Temp CoV: {cov:.6f}")

        return cov

    # -------------------------------------------------------------------------
    # 3. Scipy 최적화 실행 (Rank 0에서만)
    # -------------------------------------------------------------------------
    best_weights = None
    if comm.rank == 0:
        print(f"\n\n[Optimization] Running Physics-Informed Scipy (SLSQP)...")

        init_guess = np.ones(num_angles) / num_angles
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = [(0.0, 1.0) for _ in range(num_angles)]

        result = minimize(objective_physics, init_guess, method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 50, 'ftol': 1e-5})  # maxiter를 줄여서 시간 단축

        best_weights = result.x
        print(f"  Optimization Success: {result.success}")
        print(f"  Theoretical Best Final Temp CoV: {result.fun:.6f}")

    # -------------------------------------------------------------------------
    # 4. 결과 취합 및 스케줄 생성
    # -------------------------------------------------------------------------
    # Rank 0에서 계산된 best_weights를 모든 프로세스에 전송
    best_weights = comm.bcast(best_weights, root=0)

    optimized_schedule = []
    if comm.rank == 0: print("\n[Optimization] Final Schedule (Physics-Informed Scipy):")

    for idx, weight in enumerate(best_weights):
        duration = weight * total_steps
        if duration > 1e-3:
            fname = schedule[sorted_times[idx]]
            if comm.rank == 0:
                ang = fname.split('_')[-1].replace('.xdmf', '')
                print(f"  - Angle {ang:>3} deg: {duration:.4f} seconds")
            optimized_schedule.append((fname, duration))

    return optimized_schedule

import shutil
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        cache_dir = os.path.expanduser("~/.cache/fenics")
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"[Main] Cleared FEniCSx cache at {cache_dir}")
            except Exception as e:
                print(f"[Main] Warning: Could not clear cache: {e}")

    comm.Barrier()

    reflection_results = []
    for i in range(0, 351, 10):
        geo_file = f"../cal_by_gmsh3/microwave_geometry_and_stackdumpling_{i}.geo"
        result_file = os.path.splitext(geo_file)[0] + ".xdmf"

        if not os.path.exists(geo_file):
            if comm.rank == 0:
                print(f"Error: '{geo_file}' not found.")
            sys.exit(1)

        print(f"[Main] Loading mesh from {geo_file}...")
        domain, cell_tags, facet_tags, marker_map = create_mesh_from_geo(comm, geo_file,mesh_min=0.01,mesh_max=0.04)

        if comm.rank == 0:
            print("[Main] Physical Groups found:", marker_map)

        P_complex, s11, E_vec = solve_electromagnetic_problem2(comm, domain, cell_tags, facet_tags, marker_map, result_file=result_file)
        if comm.rank == 0:
            reflection_results.append({
                "angle": i,
                "P_real": -P_complex.real,
                "P_imag": -P_complex.imag,
                "S11_dB": s11,
                "Ex_avg_real": E_vec[0].real, "Ex_avg_imag": E_vec[0].imag,
                "Ey_avg_real": E_vec[1].real, "Ey_avg_imag": E_vec[1].imag,
                "Ez_avg_real": E_vec[2].real, "Ez_avg_imag": E_vec[2].imag,
            })

    if comm.rank == 0:
        csv_file = "result3/reflection_results.csv"
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        fieldnames = ["angle", "P_real", "P_imag", "S11_dB", 
                      "Ex_avg_real", "Ex_avg_imag", "Ey_avg_real", "Ey_avg_imag", "Ez_avg_real", "Ez_avg_imag"]
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(reflection_results)
        print(f"[Main] Saved reflection results to {csv_file}")

    static_file = "cal_by_gmsh3/microwave_geometry_and_stackdumpling_0.xdmf"
    
    schedule = {
        0: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_0.xdmf",
        10: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_10.xdmf",
        20: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_20.xdmf",
        30: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_30.xdmf",
        40: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_40.xdmf",
        50: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_50.xdmf",
        60: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_60.xdmf",
        70: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_70.xdmf",
        80: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_80.xdmf",
        90: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_90.xdmf",
        100: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_100.xdmf",
        110: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_110.xdmf",
        120: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_120.xdmf",
        130: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_130.xdmf",
        140: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_140.xdmf",
        150: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_150.xdmf",
        160: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_160.xdmf",
        170: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_170.xdmf",
        180: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_180.xdmf",
        190: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_190.xdmf",
        200: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_200.xdmf",
        210: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_210.xdmf",
        220: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_220.xdmf",
        230: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_230.xdmf",
        240: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_240.xdmf",
        250: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_250.xdmf",
        260: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_260.xdmf",
        270: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_270.xdmf",
        280: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_280.xdmf",
        290: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_290.xdmf",
        300: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_300.xdmf",
        310: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_310.xdmf",
        320: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_320.xdmf",
        330: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_330.xdmf",
        340: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_340.xdmf",
        350: "cal_by_gmsh3/microwave_geometry_and_stackdumpling_350.xdmf"
    }

    rotation_center = np.loadtxt("cal_by_gmsh3/rotation_center.txt")
    
    analyze_heating_uniformity3(comm, schedule, rotation_center)

    solve_transient_heat_transfer_rotating5(
        comm,
        schedule,
        rotation_center=rotation_center,
        output_file="result3/final_temperature_normal360T.xdmf",
        dt=10.0,
        total_time=360.0,
        initial_temp=25.0
    )
