"""Shared EM post-processing: heat source, H-field, Poynting vector, S11, scaling."""
import os

import numpy as np
import petsc4py.PETSc
import ufl
from dolfinx import fem, io
from mpi4py import MPI

from microwave_sim.constants import TARGET_POWER_W, eps0, mu0, omega


def compute_heat_source(domain, eps_r, E_h):
    """Compute volumetric heat source Q from E-field."""
    V_Q = fem.functionspace(domain, ("DG", 0))
    Q = fem.Function(V_Q)
    Q.name = "HeatSource"
    Q_expr_ufl = 0.5 * omega * eps0 * (-ufl.imag(eps_r)) * ufl.real(ufl.dot(E_h, ufl.conj(E_h)))
    Q.interpolate(fem.Expression(Q_expr_ufl, V_Q.element.interpolation_points))
    return Q


def compute_h_and_poynting(domain, E_h):
    """Compute H-field and time-averaged Poynting vector."""
    V_vec = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

    H_h = fem.Function(V_vec)
    H_h.name = "H_field"
    H_ufl = (-1.0 / (1j * omega * mu0)) * ufl.curl(E_h)
    H_h.interpolate(fem.Expression(H_ufl, V_vec.element.interpolation_points))

    S_h = fem.Function(V_vec)
    S_h.name = "Poynting_Vector"
    S_ufl = 0.5 * ufl.real(ufl.cross(E_h, ufl.conj(H_h)))
    S_h.interpolate(fem.Expression(S_ufl, V_vec.element.interpolation_points))
    return H_h, S_h, S_ufl


def interpolate_e_lagrange(domain, E_h):
    """Interpolate N1curl E-field to vector Lagrange for visualization."""
    V_lagrange = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    E_h_lagrange = fem.Function(V_lagrange)
    E_h_lagrange.name = "E_field_Lagrange"
    E_h_lagrange.interpolate(fem.Expression(E_h, V_lagrange.element.interpolation_points))
    return E_h_lagrange


def compute_s11_and_port_vector(comm, domain, E_h, E_inc, facet_tags, port_id, S_ufl, n):
    """Compute S11 (dB), complex S11, net port power, and average E-vector."""
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    flux_term = ufl.dot(S_ufl, n)

    P_in_net = 0.0
    S11_dB = 0.0
    S11_complex = 0.0 + 0.0j
    E_vec_avg = np.zeros(3, dtype=np.complex128)

    try:
        P_flux_form = fem.form(flux_term * ds(port_id))
        P_net_out = comm.allreduce(fem.assemble_scalar(P_flux_form), op=MPI.SUM)
        P_in_net = -P_net_out

        Z0 = 376.73
        E_inc_sq = ufl.inner(E_inc, E_inc)
        P_inc_form = fem.form((1.0 / (2.0 * Z0)) * E_inc_sq * ds(port_id))
        P_inc = comm.allreduce(fem.assemble_scalar(P_inc_form), op=MPI.SUM)
        P_ref = P_inc - P_in_net

        if P_inc > 1e-12:
            ratio = max(P_ref / P_inc, 0.0)
            s11_mag = np.sqrt(ratio)
            S11_dB = 20 * np.log10(s11_mag + 1e-16)

            e_tot_form = fem.form(E_h[1] * ds(port_id))
            e_inc_form = fem.form(E_inc[1] * ds(port_id))
            val_tot = comm.allreduce(fem.assemble_scalar(e_tot_form), op=MPI.SUM)
            val_inc = comm.allreduce(fem.assemble_scalar(e_inc_form), op=MPI.SUM)
            if abs(val_inc) > 1e-12:
                complex_ratio = (val_tot / val_inc) - 1.0
                S11_complex = s11_mag * np.exp(1j * np.angle(complex_ratio))
            else:
                S11_complex = s11_mag + 0j
    except Exception as e:
        if comm.rank == 0:
            print(f"[Warning] S11 calc failed: {e}")

    try:
        area_form = fem.form(fem.Constant(domain, petsc4py.PETSc.ScalarType(1.0)) * ds(port_id))
        port_area = comm.allreduce(fem.assemble_scalar(area_form), op=MPI.SUM)
        if port_area > 1e-9:
            for i in range(3):
                val = comm.allreduce(fem.assemble_scalar(fem.form(E_h[i] * ds(port_id))), op=MPI.SUM)
                E_vec_avg[i] = val / port_area
    except Exception:
        pass

    return S11_dB, S11_complex, P_in_net, E_vec_avg


def scale_to_target_power(comm, Q, E_h, H_h, S_h, E_h_lagrange, subdomains, mass_tag, target_power=TARGET_POWER_W):
    """Scale fields so absorbed power in mass region matches target."""
    dx_sub = ufl.Measure("dx", domain=Q.function_space.mesh, subdomain_data=subdomains)
    power_form = fem.form(Q * dx_sub(mass_tag))
    total_absorbed = comm.allreduce(fem.assemble_scalar(power_form), op=MPI.SUM)

    scale_factor = 1.0
    if total_absorbed > 1e-12:
        scale_factor = target_power / total_absorbed
        if comm.rank == 0:
            print(f"[EM] Scaling: {total_absorbed:.4f} W -> {target_power} W (x{scale_factor:.4e})")
        Q.x.array[:] *= scale_factor
        S_h.x.array[:] *= scale_factor
        sqrt_scale = np.sqrt(scale_factor)
        E_h_lagrange.x.array[:] *= sqrt_scale
        H_h.x.array[:] *= sqrt_scale
    elif comm.rank == 0:
        print("[Warning] Absorbed power is zero; skipping scaling.")
    return scale_factor, total_absorbed


def save_em_results(comm, domain, subdomains, output_dir, number, E_h_lagrange, H_h, S_h, Q):
    """Write EM results to XDMF."""
    if comm.rank == 0 and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    comm.Barrier()
    output_filename = os.path.join(output_dir, f"{number}.xdmf")
    with io.XDMFFile(comm, output_filename, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(subdomains, domain.geometry)
        xdmf.write_function(E_h_lagrange)
        xdmf.write_function(H_h)
        xdmf.write_function(S_h)
        xdmf.write_function(Q)
    return output_filename


def append_s_parameters_csv(output_dir, number, freq, S11_complex, S11_dB, E_vec_avg, P_in_net, scale_factor=1.0):
    """Append S-parameter row to CSV (rank 0 only)."""
    if MPI.COMM_WORLD.rank != 0:
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_path = os.path.join(output_dir, "s_parameters.csv")
    file_exists = os.path.isfile(csv_path)
    sqrt_scale = np.sqrt(scale_factor) if scale_factor != 1.0 else 1.0
    E_scaled = E_vec_avg * sqrt_scale
    with open(csv_path, "a") as f:
        if not file_exists:
            f.write(
                "Sim_Number,Freq_Hz,S11_Real,S11_Imag,S11_Mag_dB,S11_Phase_Rad,"
                "Ex_Real,Ex_Imag,Ey_Real,Ey_Imag,Ez_Real,Ez_Imag,Net_Power_Watt\n"
            )
        s11_phase = np.angle(S11_complex)
        f.write(
            f"{number},{freq},{S11_complex.real:.6e},{S11_complex.imag:.6e},{S11_dB:.6f},{s11_phase:.6f},"
            f"{E_scaled[0].real:.6e},{E_scaled[0].imag:.6e},"
            f"{E_scaled[1].real:.6e},{E_scaled[1].imag:.6e},"
            f"{E_scaled[2].real:.6e},{E_scaled[2].imag:.6e},"
            f"{P_in_net * scale_factor:.6e}\n"
        )
