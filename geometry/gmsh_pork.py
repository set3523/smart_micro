"""Gmsh geometry generation for pork-chamber microwave simulation."""
import gmsh
import numpy as np
from dolfinx.io import gmsh as dolfinx_gmsh

from microwave_sim.constants import CHAMBER_SIZE


def _add_tetrahedron_occ(vertices, offset):
    """Create a tetrahedron volume in Gmsh OCC kernel."""
    p_tags = []
    for v in vertices:
        x, y, z = v[0] + offset[0], v[1] + offset[1], v[2] + offset[2]
        p_tags.append(gmsh.model.occ.addPoint(x, y, z))

    face_indices = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
    face_tags = []
    for idxs in face_indices:
        l1 = gmsh.model.occ.addLine(p_tags[idxs[0]], p_tags[idxs[1]])
        l2 = gmsh.model.occ.addLine(p_tags[idxs[1]], p_tags[idxs[2]])
        l3 = gmsh.model.occ.addLine(p_tags[idxs[2]], p_tags[idxs[0]])
        wire = gmsh.model.occ.addCurveLoop([l1, l2, l3])
        face = gmsh.model.occ.addPlaneSurface([wire])
        face_tags.append(face)

    f_dimtags = [(2, t) for t in face_tags]
    out, _ = gmsh.model.occ.fragment(f_dimtags, [])
    new_f_tags = [t[1] for t in out if t[0] == 2]
    shell = gmsh.model.occ.addSurfaceLoop(new_f_tags)
    return gmsh.model.occ.addVolume([shell])


def generate_microwave_mesh(comm, model_rank, n_elem_x, n_elem_y, n_elem_z):
    """
    Gmsh API로 전자레인지 챔버 + pork 형상 메쉬를 생성하고 DOLFINx mesh를 반환합니다.
    Returns: (domain, cell_tags, facet_tags)
    """
    gmsh.initialize()

    if comm.rank == model_rank:
        chamber_size = list(CHAMBER_SIZE)
        dx = chamber_size[0] / n_elem_x
        dy = chamber_size[1] / n_elem_y
        dz = chamber_size[2] / n_elem_z
        lc_min = min(dx, dy, dz)
        mesh_tol = 0.0

        base_radius = 0.04
        eff_radius = base_radius + mesh_tol
        cube_side = 0.09
        eff_cube_half = cube_side / 2.0
        eff_cube_full = eff_cube_half * 2.0
        y_center_bottom = 0.08
        y_center_top = 0.18
        y_center_cube = 0.19

        chamber_tag = gmsh.model.occ.addBox(0, 0, 0, *chamber_size)

        port_x, port_y, port_z = 0.0, 0.26, 0.08
        port_dx, port_dz = 0.31, 0.15
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
        for cx, cy, cz in [
            (0.105, y_center_bottom, 0.105),
            (0.205, y_center_bottom, 0.205),
            (0.105, y_center_top, 0.119),
            (0.105, y_center_top, 0.200),
        ]:
            pork_tags.append(gmsh.model.occ.addSphere(cx, cy, cz, eff_radius))

        for cx, cz in [(0.205, 0.105), (0.205, 0.205)]:
            pork_tags.append(gmsh.model.occ.addBox(
                cx - eff_cube_half, y_center_cube - eff_cube_half, cz - eff_cube_half,
                eff_cube_full, eff_cube_full, eff_cube_full,
            ))

        H = 2.0 * eff_radius
        r_base = H / np.sqrt(2)
        v_top = np.array([0.0, eff_radius, 0.0])
        v_b1 = np.array([r_base, -eff_radius, 0.0])
        v_b2 = np.array([r_base * np.cos(2 * np.pi / 3), -eff_radius, r_base * np.sin(2 * np.pi / 3)])
        v_b3 = np.array([r_base * np.cos(4 * np.pi / 3), -eff_radius, r_base * np.sin(4 * np.pi / 3)])
        verts_up = [v_top, v_b1, v_b2, v_b3]
        verts_down = [v * np.array([1.0, -1.0, 1.0]) for v in verts_up]

        pork_tags.append(_add_tetrahedron_occ(verts_up, (0.105, y_center_bottom, 0.205)))
        pork_tags.append(_add_tetrahedron_occ(verts_down, (0.205, y_center_bottom, 0.105)))

        gmsh.model.occ.synchronize()

        chamber_dimtag = [(3, chamber_tag)]
        pork_dimtags = [(3, t) for t in pork_tags]
        tool_dimtags = pork_dimtags + [(2, port_surface_tag)]
        gmsh.model.occ.fragment(chamber_dimtag, tool_dimtags)
        gmsh.model.occ.synchronize()
        gmsh.model.geo.synchronize()

        pork_id, air_id, port_id, wall_id = 1, 2, 3, 4

        all_volumes = gmsh.model.getEntities(3)
        air_volumes, pork_volumes = [], []
        for dim, tag in all_volumes:
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            if abs(com[0] - 0.155) < 0.1 and abs(com[1] - 0.14) < 0.1:
                bbox = gmsh.model.getBoundingBox(dim, tag)
                if bbox[3] - bbox[0] < 0.30:
                    pork_volumes.append(tag)
                else:
                    air_volumes.append(tag)
            else:
                air_volumes.append(tag)

        if pork_volumes:
            gmsh.model.addPhysicalGroup(3, pork_volumes, pork_id, name="Pork")
        if air_volumes:
            gmsh.model.addPhysicalGroup(3, air_volumes, air_id, name="Air")

        all_surfaces = gmsh.model.getEntities(2)
        port_surfaces, wall_surfaces = [], []
        tol = 1e-4
        target_y = 0.26
        target_x_min, target_x_max = 0.11, 0.2
        target_z_min, target_z_max = 0.1325, 0.1775
        chamber_min = [0.0, 0.0, 0.0]
        chamber_max = chamber_size

        for dim, tag in all_surfaces:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            cx = (bbox[0] + bbox[3]) / 2.0
            cz = (bbox[2] + bbox[5]) / 2.0
            is_port = False

            if abs(bbox[1] - target_y) < tol and abs(bbox[4] - target_y) < tol:
                if (target_x_min - tol <= cx <= target_x_max + tol) and \
                        (target_z_min - tol <= cz <= target_z_max + tol):
                    if (bbox[3] - bbox[0]) > 0.01 and (bbox[5] - bbox[2]) > 0.01:
                        port_surfaces.append(tag)
                        is_port = True

            if not is_port:
                on_boundary = False
                if abs(bbox[0] - chamber_min[0]) < tol or abs(bbox[3] - chamber_max[0]) < tol:
                    on_boundary = True
                elif abs(bbox[1] - chamber_min[1]) < tol or abs(bbox[4] - chamber_max[1]) < tol:
                    on_boundary = True
                elif abs(bbox[2] - chamber_min[2]) < tol or abs(bbox[5] - chamber_max[2]) < tol:
                    on_boundary = True
                if on_boundary:
                    wall_surfaces.append(tag)

        if port_surfaces:
            gmsh.model.addPhysicalGroup(2, port_surfaces, port_id, name="port")
        if wall_surfaces:
            gmsh.model.addPhysicalGroup(2, wall_surfaces, wall_id, name="wall")

        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_min)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.mesh.clear()
        gmsh.model.mesh.generate(3)

        element_types, element_tags, node_tags = gmsh.model.mesh.getElements(3)
        if not element_types or len(element_tags[0]) == 0:
            print("[ERROR] Mesh generation failed - no 3D elements created!")
        else:
            print(f"[Mesh] Generated {len(node_tags[0])} elements in 3D")

    model_output = dolfinx_gmsh.model_to_mesh(gmsh.model, comm, rank=model_rank, gdim=3)
    domain, cell_tags, facet_tags = model_output[0], model_output[1], model_output[2]
    gmsh.finalize()
    return domain, cell_tags, facet_tags
