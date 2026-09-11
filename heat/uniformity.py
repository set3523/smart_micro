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

    ref_file = schedule[ref_time]
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
def analyze_heating_uniformity3(comm, schedule, rotation_center):
    """
    [수정됨] 전역 변수 ref_file을 사용하여 기준 메쉬를 로드합니다.
    """
    import h5py

    ref_file = get_ref_file()
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
