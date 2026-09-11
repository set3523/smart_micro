"""EM simulation on uniform structured box mesh (coordinate-based materials)."""
import pprint

import numpy as np
import ufl
from dolfinx import fem, mesh, io
from dolfinx.mesh import create_box, CellType
import dolfinx.fem.petsc
from mpi4py import MPI

from microwave_sim.constants import (
    CHAMBER_SIZE, EPS_R_AIR, EPS_R_PORK, MU_R_AIR, MU_R_PORK, freq, k0,
)
from microwave_sim.em.postprocess import (
    append_s_parameters_csv,
    compute_h_and_poynting,
    compute_heat_source,
    compute_s11_and_port_vector,
    interpolate_e_lagrange,
    save_em_results,
    scale_to_target_power,
)
from microwave_sim.geometry.pork import compute_mesh_tolerance, make_pork_region


def run_microwave_simulation(
    number="checkcheck",
    n_elem_x=93,
    n_elem_y=84,
    n_elem_z=93,
    output_dir="result",
    comm=MPI.COMM_WORLD,
    PortLA=1,
    PortLB=1,
    PortTA=1,
    PortTB=1,
    degree=1,
    petsc_options_em=None,
    petsc_options_prefix_em="test",
    cell_hex_tet="hex",
):
    """Run EM simulation on uniform box mesh with coordinate-based pork region."""
    from microwave_sim.config.petsc import gmres_gamg_options

    if petsc_options_em is None:
        petsc_options_em = gmres_gamg_options()

    rank = comm.rank
    if rank == 0:
        pprint.pprint(petsc_options_em)

    if rank == 0:
        print(f"[Uniform EM] Creating mesh ({n_elem_x}x{n_elem_y}x{n_elem_z})...")

    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array(list(CHAMBER_SIZE))
    cell_type = CellType.hexahedron if cell_hex_tet == "hex" else CellType.tetrahedron
    domain = create_box(comm, [p0, p1], [n_elem_x, n_elem_y, n_elem_z], cell_type=cell_type)

    pork_id, air_id = 1, 2
    port_id = 6

    V = fem.functionspace(domain, ("N1curl", degree))
    V_mat = fem.functionspace(domain, ("DG", 0))

    mesh_tol = compute_mesh_tolerance(n_elem_x, n_elem_y, n_elem_z)
    pork_region = make_pork_region(mesh_tol)

    tdim = domain.topology.dim
    all_cells_idx = np.arange(
        domain.topology.index_map(tdim).size_local + domain.topology.index_map(tdim).num_ghosts,
        dtype=np.int32,
    )
    pork_cells = mesh.locate_entities(domain, tdim, pork_region)
    air_cells_mask = np.ones(len(all_cells_idx), dtype=bool)
    if pork_cells.size > 0:
        air_cells_mask[pork_cells] = False
    air_cells = all_cells_idx[air_cells_mask]

    eps_r = fem.Function(V_mat)
    eps_r.name = "epsilon_r"
    eps_r.x.array[air_cells] = EPS_R_AIR
    eps_r.x.array[pork_cells] = EPS_R_PORK

    mu_r = fem.Function(V_mat)
    mu_r.name = "mu_r"
    mu_r.x.array[:] = MU_R_AIR

    markers = np.full(len(all_cells_idx), air_id, dtype=np.int32)
    markers[pork_cells] = pork_id
    subdomains = mesh.meshtags(domain, tdim, all_cells_idx, markers)

    fdim = domain.topology.dim - 1

    def left_wall(x): return np.isclose(x[0], 0.0)
    def right_wall(x): return np.isclose(x[0], CHAMBER_SIZE[0])
    def bottom_wall(x): return np.isclose(x[1], 0.0)
    def top_wall(x): return np.isclose(x[1], CHAMBER_SIZE[1])
    def back_wall(x): return np.isclose(x[2], 0.0)
    def front_wall(x): return np.isclose(x[2], CHAMBER_SIZE[2])

    pec_facets = np.concatenate([
        mesh.locate_entities_boundary(domain, fdim, right_wall),
        mesh.locate_entities_boundary(domain, fdim, back_wall),
        mesh.locate_entities_boundary(domain, fdim, front_wall),
        mesh.locate_entities_boundary(domain, fdim, left_wall),
        mesh.locate_entities_boundary(domain, fdim, bottom_wall),
        mesh.locate_entities_boundary(domain, fdim, top_wall),
    ])
    bc_pec = fem.dirichletbc(fem.Function(V), fem.locate_dofs_topological(V, fdim, pec_facets))

    portTS_z_min, portTS_z_max = 0.08, 0.23
    portTL_x_min, portTL_x_max = 0.0, 0.31
    E0 = 20000.0

    def portt_region(x):
        on_wall = np.isclose(x[1], CHAMBER_SIZE[1])
        in_z = np.logical_and(x[2] >= portTS_z_min, x[2] <= portTS_z_max)
        in_x = np.logical_and(x[0] >= portTL_x_min, x[0] <= portTL_x_max)
        return np.logical_and(on_wall, np.logical_and(in_x, in_z))

    portt_facets = mesh.locate_entities_boundary(domain, fdim, portt_region)
    ft_indices = np.concatenate([pec_facets, portt_facets])
    ft_values = np.concatenate([
        np.full(len(pec_facets), 3, dtype=np.int32),
        np.full(len(portt_facets), port_id, dtype=np.int32),
    ])
    sorted_args = np.argsort(ft_indices)
    facet_tags = mesh.meshtags(domain, fdim, ft_indices[sorted_args], ft_values[sorted_args])

    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    n = ufl.FacetNormal(domain)

    def port_field_func_T(x):
        values = np.zeros((3, x.shape[1]), dtype=np.complex128)
        width_long = portTL_x_max - portTL_x_min
        values[PortTA, :] = E0 * np.sin(np.pi * (x[PortTB] - portTL_x_min) / width_long)
        return values

    E_inc = fem.Function(V)
    E_inc.interpolate(port_field_func_T)

    E, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(1 / mu_r * ufl.curl(E), ufl.curl(v)) * ufl.dx - k0 ** 2 * ufl.inner(eps_r * E, v) * ufl.dx
    L = ufl.inner(fem.Constant(domain, np.array([0, 0, 0], dtype=np.complex128)), v) * ufl.dx
    a += 1j * k0 * ufl.inner(ufl.cross(n, E), ufl.cross(n, v)) * ds(port_id)
    L += 2 * 1j * k0 * ufl.inner(ufl.cross(n, E_inc), ufl.cross(n, v)) * ds(port_id)

    E_h = fem.Function(V)
    E_h.name = "E_field"
    try:
        problem = fem.petsc.LinearProblem(
            a, L, bcs=[bc_pec], u=E_h,
            petsc_options=petsc_options_em,
            petsc_options_prefix=petsc_options_prefix_em,
        )
        problem.solve()
        if rank == 0:
            print(f"[Uniform EM] Solver converged in {problem.solver.getIterationNumber()} iterations.")
    except Exception as e:
        if rank == 0:
            print(f"[Error] Solver failed: {e}")
        return None, None, None

    Q = compute_heat_source(domain, eps_r, E_h)
    H_h, S_h, S_ufl = compute_h_and_poynting(domain, E_h)
    E_h_lagrange = interpolate_e_lagrange(domain, E_h)

    S11_dB, S11_complex, P_in_net, E_vec_avg = compute_s11_and_port_vector(
        comm, domain, E_h, E_inc, facet_tags, port_id, S_ufl, n,
    )
    scale_factor, _ = scale_to_target_power(
        comm, Q, E_h, H_h, S_h, E_h_lagrange, subdomains, pork_id,
    )
    P_in_net *= scale_factor

    append_s_parameters_csv(output_dir, number, freq, S11_complex, S11_dB, E_vec_avg, P_in_net, scale_factor)
    save_em_results(comm, domain, subdomains, output_dir, number, E_h_lagrange, H_h, S_h, Q)

    if rank == 0:
        print(f"[Debug] S11: {S11_dB:.2f} dB, Port E: {E_vec_avg}")

    return E_h_lagrange, Q, E_vec_avg
