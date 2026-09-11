"""Memory reporting utilities."""
import resource
from mpi4py import MPI


def print_peak_memory(comm=None):
    """Print peak memory usage across MPI ranks."""
    if comm is None:
        comm = MPI.COMM_WORLD

    peak_mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mem_gb = peak_mem_kb / (1024 * 1024)

    total_peak_gb = comm.reduce(peak_mem_gb, op=MPI.SUM, root=0)
    max_single_peak_gb = comm.reduce(peak_mem_gb, op=MPI.MAX, root=0)

    if comm.rank == 0:
        print("\n[Peak Memory Report]")
        print(f" - 전체 코어 합산 피크 메모리: {total_peak_gb:.2f} GB")
        print(f" - 단일 프로세스 최대 피크: {max_single_peak_gb:.2f} GB")
        print("-" * 30)
