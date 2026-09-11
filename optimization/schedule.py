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

def find_optimal_schedule_greedy(comm, schedule, rotation_center, total_steps=35):
    """
    그리디 알고리즘을 사용하여 목표 시간(total_steps) 동안
    가열 균일도(CoV)를 최소화하는 최적의 각도 조합(스케줄)을 찾습니다.
    """
    import h5py

    ref_file = get_ref_file()
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
    ref_file = get_ref_file()
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

    ref_file = get_ref_file()
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
    ref_file = get_ref_file()
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
