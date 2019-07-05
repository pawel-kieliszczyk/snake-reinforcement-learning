import os, subprocess, sys
from mpi4py import MPI


def mpi_fork(num_of_processes):
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREAD="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-n", str(num_of_processes)]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


def get_process_id():
    return MPI.COMM_WORLD.rank


def broadcast(data):
    MPI.COMM_WORLD.Bcast(data, root=0)


def send(data, dest):
    MPI.COMM_WORLD.Send(data, dest=dest)


def receive(data, source):
    MPI.COMM_WORLD.Recv(data, source=source)

