from microwave_sim.em.geo_solver import solve_electromagnetic_problem, solve_electromagnetic_problem2
from microwave_sim.em.uniform_solver import run_microwave_simulation as run_uniform_em
from microwave_sim.em.gmsh_pork_solver import run_microwave_simulation as run_gmsh_pork_em

__all__ = [
    "solve_electromagnetic_problem",
    "solve_electromagnetic_problem2",
    "run_uniform_em",
    "run_gmsh_pork_em",
]
