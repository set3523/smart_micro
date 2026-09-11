"""PETSc solver option presets."""
import os


def lu_mumps_options(tmpdir=None):
    """Direct LU solver with MUMPS (recommended for medium meshes)."""
    return {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_22": 1,
        "mat_mumps_ooc_tmpdir": tmpdir or os.getcwd(),
        "mat_mumps_icntl_14": 200,
        "ksp_view": None,
        "ksp_converged_reason": None,
    }


def gmres_gamg_options():
    """Iterative GMRES with GAMG preconditioner."""
    return {
        "ksp_error_if_not_converged": True,
        "ksp_type": "gmres",
        "pc_type": "gamg",
        "ksp_rtol": 1e-8,
        "ksp_atol": 1e-12,
        "ksp_max_it": 1000,
        "ksp_view": None,
        "ksp_monitor": None,
        "ksp_gmres_restart": 200,
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "pc_gamg_coarse_eq_dof": 3,
        "pc_gamg_symm": "false",
        "pc_gamg_cycle_type": "w",
        "pc_gamg_threshold": 0.1,
    }
