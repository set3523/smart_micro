"""Pork/food region definitions for uniform-box mesh simulations."""
import numpy as np

from microwave_sim.constants import CHAMBER_SIZE


def compute_mesh_tolerance(n_elem_x, n_elem_y, n_elem_z):
    """Cell-center to vertex distance tolerance for conservative region selection."""
    dx = CHAMBER_SIZE[0] / n_elem_x
    dy = CHAMBER_SIZE[1] / n_elem_y
    dz = CHAMBER_SIZE[2] / n_elem_z
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) / 2.0


def _is_inside_convex(points, vertices):
    """Return mask for points inside a convex polyhedron defined by vertices."""
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
        if norm_len == 0:
            continue
        normal = normal / norm_len
        if np.dot(normal, centroid - p0) > 0:
            normal = -normal
        diff = points - p0[:, np.newaxis]
        dot_product = np.einsum("ij, i -> j", diff, normal)
        inside = np.logical_and(inside, dot_product <= 0)
    return inside


def make_pork_region(mesh_tol):
    """
    Build a coordinate predicate for pork cells (spheres, tetrahedra, cubes).
    Returns a callable suitable for dolfinx mesh.locate_entities.
    """
    base_radius = 0.04
    eff_radius = base_radius + mesh_tol

    y_center_bottom = 0.08
    y_center_top = 0.18

    cube_side = 0.09
    cube_half = cube_side / 2.0
    eff_cube_half = cube_half + mesh_tol
    y_center_cube = 0.19

    sphere_configs = [
        (0.105, y_center_bottom, 0.105),
        (0.205, y_center_bottom, 0.205),
        (0.105, y_center_top, 0.119),
        (0.105, y_center_top, 0.200),
    ]

    H = 2.0 * eff_radius
    r_base = H / np.sqrt(2)
    v_top = np.array([0.0, eff_radius, 0.0])
    v_b1 = np.array([r_base, -eff_radius, 0.0])
    v_b2 = np.array([r_base * np.cos(2 * np.pi / 3), -eff_radius, r_base * np.sin(2 * np.pi / 3)])
    v_b3 = np.array([r_base * np.cos(4 * np.pi / 3), -eff_radius, r_base * np.sin(4 * np.pi / 3)])
    verts_up = [v_top, v_b1, v_b2, v_b3]
    verts_down = [v * np.array([1.0, -1.0, 1.0]) for v in verts_up]

    cube_centers = [(0.205, 0.105), (0.205, 0.205)]

    def pork_region(x):
        in_spheres = np.zeros_like(x[0], dtype=bool)
        for cx, cy, cz in sphere_configs:
            dist_sq = (x[0] - cx) ** 2 + (x[1] - cy) ** 2 + (x[2] - cz) ** 2
            in_spheres = np.logical_or(in_spheres, dist_sq <= eff_radius ** 2)

        cx1, cz1 = 0.105, 0.205
        local_points_1 = np.array([x[0] - cx1, x[1] - y_center_bottom, x[2] - cz1])
        in_tetra_up = _is_inside_convex(local_points_1, np.array(verts_up))

        cx2, cz2 = 0.205, 0.105
        local_points_2 = np.array([x[0] - cx2, x[1] - y_center_bottom, x[2] - cz2])
        in_tetra_down = _is_inside_convex(local_points_2, np.array(verts_down))

        in_cubes = np.zeros_like(x[0], dtype=bool)
        for cx, cz in cube_centers:
            in_x = np.logical_and(x[0] >= cx - eff_cube_half, x[0] <= cx + eff_cube_half)
            in_y = np.logical_and(x[1] >= y_center_cube - eff_cube_half, x[1] <= y_center_cube + eff_cube_half)
            in_z = np.logical_and(x[2] >= cz - eff_cube_half, x[2] <= cz + eff_cube_half)
            in_cubes = np.logical_or(in_cubes, np.logical_and(in_x, np.logical_and(in_y, in_z)))

        return np.logical_or(
            in_spheres,
            np.logical_or(np.logical_or(in_tetra_up, in_tetra_down), in_cubes),
        )

    return pork_region
