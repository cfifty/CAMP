from hyperopt import hp
import hyperopt
from multiprocessing import Pool
import multiprocessing
import os

# TODO(cfifty): kill everything: fuser -v /dev/nvidia* | grep '3' | xargs kill
EMBEDDINGS_SEARCH = {'topology': [0, 1, 2, 3], 'lm': [4, 5, 6, 7]}
GPU_TO_EMB = {0: 'topology', 1: 'topology', 2: 'topology', 3: 'topology', 4: 'topology', 5: 'topology', 6: 'topology', 7: 'lm'}

"""
python transformer_hparam.py
"""

def train_model(q, space):
    gpu_id = q.get()
    try:
        # Add non-searchable args.
        space['--cuda'] = f'{gpu_id}'
        space['--save-dir'] = f'{GPU_TO_EMB[gpu_id]}_hparam_search'
        space['--task-list-file'] = 'datasets/simulation_data.json'
        space['--num_epochs'] = '101'
        space['--model_size'] = 'base'
        space['--position_embeddings'] = GPU_TO_EMB[gpu_id]
        space['--batch_size'] = '1024'

        # Build the command.
        cmd = 'python transformer_train.py ../massive_simulation_dataset'
        for key in space:
            cmd = f'{cmd} {key} {space[key]}'
        os.system(cmd)
    finally:
        q.put(gpu_id)


def main():
    fspace = {
        '--learning-rate': hp.choice('--learning-rate', [1e-3, 1e-4, 5e-4, 5e-5]),
        '--dropout': hp.choice('--dropout', [0.0, 0.1]),
        '--weight_decay': hp.choice('--weight_decay', [0.0, 0.03, 0.1]),
        '--warmup_steps': hp.choice('--warmup_steps', [2000, 5000, 10000])
    }
    gpus = [0, 1, 2, 3, 4, 5, 6]
    num_gpus = len(gpus)
    num_process_per_gpu = 1

    num_trials = 1000

    with Pool(num_gpus * num_process_per_gpu) as pool:
        m = multiprocessing.Manager()

        # Add each GPU number to a multi-process queue.
        q = m.Queue()
        for i in gpus:
            for _ in range(num_process_per_gpu):
                q.put(i)

        # Run an experiment as soon as the GPU returns.
        for _ in range(num_trials):
            pool.apply_async(train_model, args=(q, hyperopt.pyll.stochastic.sample(fspace),))
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()
