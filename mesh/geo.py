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
