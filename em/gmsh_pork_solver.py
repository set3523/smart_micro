"""EM simulation using Gmsh-generated pork-chamber mesh."""
import os
import pprint

import numpy as np
import ufl
from dolfinx import fem, io
import dolfinx.fem.petsc
from mpi4py import MPI

from microwave_sim.constants import EPS_R_AIR, EPS_R_PORK, MU_R_AIR, MU_R_PORK, freq, k0
from microwave_sim.em.postprocess import compute_heat_source, interpolate_e_lagrange
from microwave_sim.geometry.gmsh_pork import generate_microwave_mesh


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
):
    """Run EM simulation on Gmsh pork-chamber mesh."""
    from microwave_sim.config.petsc import lu_mumps_options

    if petsc_options_em is None:
        petsc_options_em = lu_mumps_options()

    rank = comm.rank
    if rank == 0:
        pprint.pprint(petsc_options_em)
        print(f"[Gmsh EM] Generating mesh ({n_elem_x}x{n_elem_y}x{n_elem_z})...")

    domain, cell_tags, facet_tags = generate_microwave_mesh(comm, 0, n_elem_x, n_elem_y, n_elem_z)

    mass_id, air_id, port_id, wall_id = 1, 2, 3, 4

    V = fem.functionspace(domain, ("N1curl", degree))
    V_mat = fem.functionspace(domain, ("DG", 0))

    eps_r = fem.Function(V_mat)
    eps_r.name = "epsilon_r"
    mu_r = fem.Function(V_mat)
    mu_r.name = "mu_r"
    eps_r.x.array[:] = EPS_R_AIR
    mu_r.x.array[:] = MU_R_AIR

    mass_cells = cell_tags.indices[cell_tags.values == mass_id]
    if len(mass_cells) > 0:
        eps_r.x.array[mass_cells] = EPS_R_PORK
        mu_r.x.array[mass_cells] = MU_R_PORK

    subdomains = cell_tags
    fdim = domain.topology.dim - 1
    wall_facets = facet_tags.indices[facet_tags.values == wall_id]

    bcs = []
    if len(wall_facets) > 0:
        wall_dofs = fem.locate_dofs_topological(V, fdim, wall_facets)
        u_pec = fem.Function(V)
        u_pec.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(u_pec, wall_dofs))

    port_facets = facet_tags.indices[facet_tags.values == port_id]
    portTL_x_min, portTL_x_max = 0.0, 0.31
    E0 = 20000.0

    def port_field_func_T(x):
        values = np.zeros((3, x.shape[1]), dtype=np.complex128)
        width_long = portTL_x_max - portTL_x_min
        values[PortTA, :] = E0 * np.sin(np.pi * (x[PortTB] - portTL_x_min) / width_long)
        return values

    E_inc = fem.Function(V)
    E_inc.interpolate(port_field_func_T)

    E, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    n = ufl.FacetNormal(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    a = ufl.inner(1 / mu_r * ufl.curl(E), ufl.curl(v)) * ufl.dx - k0 ** 2 * ufl.inner(eps_r * E, v) * ufl.dx
    L = ufl.inner(fem.Constant(domain, np.array([0, 0, 0], dtype=np.complex128)), v) * ufl.dx

    if len(port_facets) > 0:
        a += 1j * k0 * ufl.inner(ufl.cross(n, E), ufl.cross(n, v)) * ds(port_id)
        L += 2 * 1j * k0 * ufl.inner(ufl.cross(n, E_inc), ufl.cross(n, v)) * ds(port_id)

    E_h = fem.Function(V)
    E_h.name = "E_field"
    try:
        problem = fem.petsc.LinearProblem(
            a, L, bcs=bcs, u=E_h,
            petsc_options=petsc_options_em,
            petsc_options_prefix=petsc_options_prefix_em,
        )
        problem.solve()
    except Exception as e:
        if rank == 0:
            print(f"[Error] Solver failed: {e}")
        return None, None, None

    Q = compute_heat_source(domain, eps_r, E_h)
    E_h_lagrange = interpolate_e_lagrange(domain, E_h)

    E_vec_avg = np.zeros(3, dtype=np.complex128)
    try:
        if len(port_facets) > 0:
            area_form = fem.form(fem.Constant(domain, np.complex128(1.0)) * ds(port_id))
            port_area = comm.allreduce(fem.assemble_scalar(area_form), op=MPI.SUM)
            if port_area > 1e-9:
                for i in range(3):
                    val = comm.allreduce(fem.assemble_scalar(fem.form(E_h[i] * ds(port_id))), op=MPI.SUM)
                    E_vec_avg[i] = val / port_area
    except Exception as e:
        if rank == 0:
            print(f"[Warning] Port vector calc failed: {e}")

    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    comm.Barrier()
    output_filename = os.path.join(output_dir, f"{number}.xdmf")
    with io.XDMFFile(comm, output_filename, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(subdomains, domain.geometry)
        xdmf.write_meshtags(facet_tags, domain.geometry)
        xdmf.write_function(E_h_lagrange)
        xdmf.write_function(Q)

    if rank == 0:
        print(f"[Gmsh EM] Saved to {output_filename}, Port E: {E_vec_avg}")

    return E_h_lagrange, Q, E_vec_avg
