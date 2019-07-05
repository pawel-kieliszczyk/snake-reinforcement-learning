import os, multiprocessing
import tensorflow as tf

import mpi_helper, tensorflow_helper

from master_process import Master
from worker_process import Worker


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NUM_OF_WORKERS = multiprocessing.cpu_count()
NUM_OF_PROCESSES = NUM_OF_WORKERS + 1

mpi_helper.mpi_fork(NUM_OF_PROCESSES)
process_id = mpi_helper.get_process_id()

tensorflow_helper.setup_gpu_memory_usage()

training_epochs = 6 * 4000
worker_batch_size = 64

if process_id == 0:
    print("Starting master with {} workers".format(NUM_OF_WORKERS))
    master = Master()
    master.run(NUM_OF_WORKERS, training_epochs, worker_batch_size)
else:
    worker = Worker(process_id)
    worker.run(training_epochs, worker_batch_size)

