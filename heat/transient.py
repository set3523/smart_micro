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
    ref_file = get_ref_file()
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
