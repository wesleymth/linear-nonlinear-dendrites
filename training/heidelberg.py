#!/usr/bin/env python
"""Train multiple networks to recognize spoken digits.

Some code is modified from Friedemann Zenke's
SpyTorch (https://doi.org/10.5281/zenodo.3724018).

"""
import os
from copy import deepcopy
from typing import Tuple, Dict, Optional
import multiprocessing as mp
import itertools
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import gzip
import shutil
import hashlib
import urllib.request
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlretrieve
from tqdm import trange

from model_components import (
    get_spike_fn,
    get_default_dendritic_fn,
    get_sigmoid_fn,
    SpikingNetwork,
    RecurrentSpikingNetwork,
    TwoCompartmentSpikingNetwork,
    RecurrentNeuronParameters,
    ParallelSpikingNetwork,
    PRCSpikingNetwork,
    PRCNeuronParameters,
    NetworkArchitecture,
    Environment,
)
from memorize import (
    classification_accuracy as _minibatch_classification_accuracy,
)


NETWORK_ARCHITECTURE = NetworkArchitecture((700, 200, 20))
Environment.nb_steps = 100
NUM_SEEDS = 1
EPOCHS = 200
SWEEP_DURATION = 1.4

CACHE_DIR = os.path.expanduser("/tmp/lnl-dendrite-data")
CACHE_SUBDIR = "hdspikes"

NUM_WORKERS = 1
NUM_THREADS = 4

class Data:
    def __init__(self, path_to_train_data: str, path_to_test_data: str):
        self._path_to_train = path_to_train_data
        self._path_to_test = path_to_test_data

        self._train_file: Optional[h5py.File] = None
        self._test_file: Optional[h5py.File] = None

        self.x_train: Optional[h5py.Dataset] = None
        self.y_train: Optional[h5py.Dataset] = None
        self.x_test: Optional[h5py.Dataset] = None
        self.y_test: Optional[h5py.Dataset] = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *err_args):
        self.close()

    def open(self):
        """Open connections to the training and test data files."""
        self._train_file = h5py.File(self._path_to_train, 'r')
        self.x_train = self._train_file['spikes']
        self.y_train = self._train_file['labels']

        self._test_file = h5py.File(self._path_to_test, 'r')
        self.x_test = self._test_file['spikes']
        self.y_test = self._test_file['labels']

    def close(self):
        """Close connections to training and test data files."""
        self._train_file.close()
        self._test_file.close()
        for a, b in itertools.product(['x', 'y'], ['train', 'test']):
            setattr(self, '_'.join((a, b)), None)

class DefaultOptimizer:
    def __init__(self, forward_fn, params):
        self._forward = forward_fn
        self.params = params
        self.optimizer = torch.optim.Adamax(
            params, lr=2e-3, betas=(0.9, 0.999)
        )

        log_softmax_fn = nn.LogSoftmax(dim=1)
        neg_log_lik_fn = nn.NLLLoss()

        def loss_fn(actual_output, desired_output):
            m, _ = torch.max(actual_output, 1)
            log_p_y = log_softmax_fn(m)
            loss_val = neg_log_lik_fn(log_p_y, desired_output.long())
            return loss_val

        self.loss_fn = loss_fn
        self.loss_history = []
        self.accuracy_history = []  # Classification accuracy at each epoch

    def optimize(self, data_loader:DataLoader, epochs): #input_, desired_output, epochs):
        for e in trange(epochs):
            batch_loss = []
            batch_accuracy = []
            for batch_x, batch_y in data_loader:
            
            # sparse_data_generator_from_hdf5_spikes(
            #     input_, desired_output, SWEEP_DURATION, shuffle=True
            # ):
                actual_output = self._forward(batch_x.to_dense())

                self.optimizer.zero_grad()
                loss_val = self.loss_fn(actual_output, batch_y)
                loss_val.backward()
                self.optimizer.step()

                batch_loss.append(loss_val.item())
                batch_accuracy.append(self._accuracy(actual_output, batch_y))

            self.loss_history.append(np.mean(batch_loss))
            self.accuracy_history.append(np.mean(batch_accuracy))

    @staticmethod
    def _accuracy(
        actual_output: torch.Tensor, desired_output: torch.Tensor
    ) -> float:
        max_over_time, _ = torch.max(actual_output, 1)
        # argmax over output units
        _, predicted_category = torch.max(max_over_time, 1)

        accuracy = np.mean(
            (desired_output == predicted_category).detach().cpu().numpy()
        )
        return accuracy


def get_shd_dataset(cache_dir, cache_subdir):

    # The remote directory with the data files
    base_url = "https://compneuro.net/datasets"

    # Retrieve MD5 hashes from remote
    response = urllib.request.urlopen("%s/md5sums.txt" % base_url)
    data = response.read()
    lines = data.decode('utf-8').split("\n")
    file_hashes = {
        line.split()[1]: line.split()[0]
        for line in lines
        if len(line.split()) == 2
    }
    # Download the Spiking Heidelberg Digits (SHD) dataset
    files = [
        "shd_train.h5.gz",
        "shd_test.h5.gz",
    ]
    for fn in files:
        origin = "%s/%s" % (base_url, fn)
        hdf5_file_path = get_and_gunzip(
            origin,
            fn,
            md5hash=file_hashes[fn],
            cache_dir=cache_dir,
            cache_subdir=cache_subdir,
        )
        print("File %s decompressed to:" % (fn))
        print(hdf5_file_path)


def get_and_gunzip(
    origin, filename, md5hash=None, cache_dir=None, cache_subdir=None
):
    gz_file_path = get_file(
        filename,
        origin,
        md5_hash=md5hash,
        cache_dir=cache_dir,
        cache_subdir=cache_subdir,
    )
    if gz_file_path.lower().endswith('.gz'):
        hdf5_file_path = gz_file_path[:-3]
        assert hdf5_file_path.lower().endswith(('.h5', '.hdf5'))
    else:
        raise RuntimeError(
            f'Expected gz_file_path to end with .gz: {gz_file_path}'
        )
    if not os.path.isfile(hdf5_file_path) or os.path.getctime(
        gz_file_path
    ) > os.path.getctime(hdf5_file_path):
        print("Decompressing %s" % gz_file_path)
        with gzip.open(gz_file_path, 'r') as f_in, open(
            hdf5_file_path, 'wb'
        ) as f_out:
            shutil.copyfileobj(f_in, f_out)
    return hdf5_file_path


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    if (algorithm == 'sha256') or (
        algorithm == 'auto' and len(file_hash) == 64
    ):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(hash) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def get_file(
    fname,
    origin,
    md5_hash=None,
    file_hash=None,
    cache_subdir='datasets',
    hash_algorithm='auto',
    extract=False,
    archive_format='auto',
    cache_dir=None,
):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.data-cache')
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.path.exists(datadir_base):
        os.makedirs(datadir_base)
    elif not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.data-cache')
    datadir = os.path.join(datadir_base, cache_subdir)
    os.makedirs(datadir, exist_ok=True)

    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print(
                    'A local file was found, but it seems to be '
                    'incomplete or outdated because the '
                    + hash_algorithm
                    + ' file hash does not match the original value of '
                    + file_hash
                    + ' so we will re-download the data.'
                )
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)

    error_msg = 'URL fetch failure on {}: {} -- {}'
    try:
        try:
            urlretrieve(origin, fpath)
        except HTTPError as e:
            raise Exception(error_msg.format(origin, e.code, e.msg))
        except URLError as e:
            raise Exception(error_msg.format(origin, e.errno, e.reason))
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(fpath):
            os.remove(fpath)

    return fpath



# def sparse_data_generator_from_hdf5_spikes(
#     X, y, sweep_duration: float, shuffle=True,
# ):
#     """ This generator takes a spike dataset and generates spiking network input as sparse tensors.

#     Args:
#         X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
#         y: The labels
#     """

#     labels_ = np.array(y, dtype=np.int)
#     number_of_batches = len(labels_) // Environment.batch_size
#     sample_index = np.arange(len(labels_))

#     # compute discrete firing times
#     firing_times = X['times']
#     units_fired = X['units']

#     time_bins = np.linspace(0, sweep_duration, num=Environment.nb_steps)

#     if shuffle:
#         np.random.shuffle(sample_index)

#     total_batch_count = 0
#     counter = 0
#     while counter < number_of_batches:
#         batch_index = sample_index[
#             Environment.batch_size
#             * counter : Environment.batch_size
#             * (counter + 1)
#         ]

#         coo = [[] for i in range(3)]
#         for bc, idx in enumerate(batch_index):
#             times = np.digitize(firing_times[idx], time_bins)
#             units = units_fired[idx]
#             batch = [bc for _ in range(len(times))]

#             coo[0].extend(batch)
#             coo[1].extend(times)
#             coo[2].extend(units)

#         i = torch.LongTensor(coo).to(Environment.device)
#         v = torch.FloatTensor(np.ones(len(coo[0]))).to(Environment.device)

#         X_batch = torch.sparse.FloatTensor(
#             i,
#             v,
#             torch.Size(
#                 [
#                     Environment.batch_size,
#                     Environment.nb_steps,
#                     NETWORK_ARCHITECTURE.nb_units_by_layer[0],
#                 ]
#             ),
#         ).to(Environment.device)
#         y_batch = torch.tensor(labels_[batch_index], device=Environment.device)

#         yield X_batch.to(device=Environment.device), y_batch.to(
#             device=Environment.device
#         )

#         counter += 1
        
def sparse_data_generator_from_hdf5_spikes(X, y, batch_size, nb_steps, nb_units, max_time, shuffle=True):
    """ This generator takes a spike dataset and generates spiking network input as sparse tensors. 

    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels
    """
    device = Environment.device

    labels_ = np.array(y,dtype=int)
    number_of_batches = len(labels_)//batch_size
    sample_index = np.arange(len(labels_))

    # compute discrete firing times
    firing_times = X['times']
    units_fired = X['units']
    
    time_bins = np.linspace(0, max_time, num=nb_steps)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter<number_of_batches:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]

        coo = [ [] for i in range(3) ]
        for bc,idx in enumerate(batch_index):
            times = np.digitize(firing_times[idx], time_bins)
            units = units_fired[idx]
            batch = [bc for _ in range(len(times))]
            
            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)
    
        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size,nb_steps,nb_units])).to(device)
        y_batch = torch.tensor(labels_[batch_index],device=device)

        yield X_batch.to(device=device), y_batch.to(device=device)

        counter += 1
        
class SHDDataset(Dataset):
    def __init__(self, train_file_path:str, nb_steps:int, nb_units:int, max_time:float)->None:
        
        train_file = h5py.File(train_file_path, 'r')
        x_train = train_file['spikes']
        y_train = train_file['labels']
        
        # compute discrete firing times
        
        self.data = [(torch.squeeze(measure.to_dense()), label) 
                     for measure, label in 
                     sparse_data_generator_from_hdf5_spikes(x_train, y_train, 1, nb_steps, nb_units, max_time, shuffle=False)]
        
        train_file.close()

    def __len__(self)->int:
        return len(self.data)

    def __getitem__(self, idx)->Tuple[torch.Tensor, int]:
        x, y = self.data[idx]
        return x, y.item()


def main():
    """Run training loop across multiple random seeds in parallel."""
    #get_shd_dataset(CACHE_DIR, CACHE_SUBDIR)
    
    
    # print(f'Creating the training dataloader ...')
    # train_dataloader = DataLoader(training_data, batch_size=Environment.batch_size, shuffle=True, drop_last=True)
    # print(f'Creating the testing dataloader ...')
    # test_dataloader = DataLoader(testing_data, batch_size=Environment.batch_size, shuffle=False, drop_last=True)
    
    # print(f'Deleting uneccessary Dataset ...')
    # del training_data
    # del testing_data
    
    #worker(0,train_dataloader, test_dataloader)
    
    # train_dataloaders = [train_dataloader for _ in range(NUM_SEEDS)]
    # test_dataloaders = [test_dataloader for _ in range(NUM_SEEDS)]
    #torch.multiprocessing.set_sharing_strategy('file_system')
    
    # with mp.Pool(NUM_WORKERS) as pool:
    #     pool.starmap(worker, zip(range(NUM_SEEDS), itertools.repeat(deepcopy(training_data)), itertools.repeat(deepcopy(testing_data))))
    
    # def init_worker(training_dataloader, testing_dataloader):
    #     print(f'Initializing the workers ...')
    #     global training_dataloader_
    #     global testing_dataloader_
    #     # store argument in the global variable for this process
    #     training_dataloader_ = training_dataloader
    #     testing_dataloader_ = testing_dataloader
        
    # print(f'Starting multi-processing ...')
    # with mp.Pool(NUM_WORKERS, initializer=init_worker, initargs=(train_dataloader, test_dataloader)) as pool:
    #     pool.map(worker, range(NUM_SEEDS))
    
    # print(f'Starting multi-processing ...')
    
    
    with mp.Pool(NUM_WORKERS) as pool:
        pool.map(worker, range(NUM_SEEDS))
    
    #worker(0)
    
    # processes = []
    # for seed_no in range(NUM_WORKERS):
    #     p = mp.Process(target=worker, args=(seed_no, train_dataloader, test_dataloader))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
    
    # with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    #     executor.map(worker, range(NUM_SEEDS), train_dataloaders, test_dataloaders)


def worker(rep_num: int, set_seed:bool=True):
    """_summary_

    Parameters
    ----------
    rep_num : int
        _description_
    set_seed : bool, optional
        _description_, by default True
    """
    torch.set_num_threads(NUM_THREADS)
    print(f'Creating the training & testing data for seed no{rep_num}...')
    training_data = SHDDataset('./data/shd_train.h5', Environment.nb_steps, NETWORK_ARCHITECTURE.nb_units_by_layer[0], SWEEP_DURATION)
    testing_data = SHDDataset('./data/shd_test.h5', Environment.nb_steps, NETWORK_ARCHITECTURE.nb_units_by_layer[0], SWEEP_DURATION)
    if set_seed:
        torch.use_deterministic_algorithms(True)
        random.seed(rep_num)
        np.random.seed(rep_num)
        torch.manual_seed(rep_num)
        print(f'rep_num')
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        g = torch.Generator()
        g.manual_seed(rep_num)
        train_dataloader = DataLoader(training_data, 
                                    batch_size=Environment.batch_size, 
                                    shuffle=True,
                                    drop_last=True,
                                    worker_init_fn=seed_worker,
                                    generator=g)
    
        testing_dataloader = DataLoader(testing_data, 
                                        batch_size=Environment.batch_size, 
                                        shuffle=False, 
                                        drop_last=True,
                                        worker_init_fn=seed_worker,
                                        generator=g)
    else:
        train_dataloader = DataLoader(training_data, 
                                    batch_size=Environment.batch_size, 
                                    shuffle=True,
                                    drop_last=True)
    
        testing_dataloader = DataLoader(testing_data, 
                                        batch_size=Environment.batch_size, 
                                        shuffle=False, 
                                        drop_last=True)
        
    print(f'Deleting unecessary training & testing data for seed no{rep_num}...')
    del training_data
    del testing_data
    
    nets, optimizers = train_networks(rep_num, train_dataloader, testing_dataloader)
    save_loss_history(
        optimizers,
        f'heidelberg_training_results_{rep_num}.csv',
    )
    save_test_accuracy(optimizers, f'heidelberg_test_accuracy_{rep_num}.json')


def train_networks(
    rep_num: int, train_dataloader:DataLoader, test_dataloader:DataLoader, epochs: int = EPOCHS
) -> Tuple[Dict[str, SpikingNetwork], Dict[str, DefaultOptimizer]]:
    """Train a set of PRC models to memorize a random dataset."""

    nets = get_networks()
    optimizers = get_optimizers(nets)

    # path_to_train_data = os.path.join(CACHE_DIR, CACHE_SUBDIR, 'shd_train.h5')
    # path_to_test_data = os.path.join(CACHE_DIR, CACHE_SUBDIR, 'shd_test.h5')

    #with Data(path_to_train_data, path_to_test_data) as data:
    for label in nets:
        print(f'Training \"{label}\" - {rep_num}')
        # initial_train_accuracy = classification_accuracy(
        #     data.x_train, data.y_train, nets[label]
        # )
        # initial_test_accuracy = classification_accuracy(
        #     data.x_test, data.y_test, nets[label]
        # )
        # optimizers[label].optimize(
        #     data.x_train, data.y_train, epochs
        # )
        # final_train_accuracy = classification_accuracy(
        #     data.x_train, data.y_train, nets[label]
        # )
        # final_test_accuracy = classification_accuracy(
        #     data.x_test, data.y_test, nets[label]
        # )
        
        initial_train_accuracy = classification_accuracy(
            train_dataloader, nets[label]
        )
        initial_test_accuracy = classification_accuracy(
            test_dataloader, nets[label]
        )
        optimizers[label].optimize(
            train_dataloader, epochs
        )
        final_train_accuracy = classification_accuracy(
            train_dataloader, nets[label]
        )
        final_test_accuracy = classification_accuracy(
            test_dataloader, nets[label]
        )

        optimizers[label].test_accuracy = {
            'initial': float(initial_test_accuracy),
            'final': float(final_test_accuracy),
        }
        print(
            f'Finished training \"{label}\" - {rep_num}; '
            f'Initial Train Acc. {100 * initial_train_accuracy:.1f}%, '
            f'Initial Test Acc. {100 * initial_test_accuracy:.1f}%.'
            f'Final Train Acc. {100 * final_train_accuracy:.1f}%.'
            f'Final Test Acc. {100 *  final_test_accuracy:.1f}%.'
        )

    return nets, optimizers


def save_loss_history(
    optimizers: Dict[str, DefaultOptimizer], fname: str
) -> None:
    """Save loss during training to CSV file."""
    data = {'model_name': [], 'epoch': [], 'loss': [], 'accuracy': []}

    for label, optimizer in optimizers.items():
        assert len(optimizer.loss_history) == len(optimizer.accuracy_history)
        num_epochs = len(optimizer.loss_history)

        data['model_name'].extend([label] * num_epochs)
        data['epoch'].extend(range(num_epochs))
        data['loss'].extend(optimizer.loss_history)
        data['accuracy'].extend(optimizer.accuracy_history)

    data_df = pd.DataFrame(data)
    data_df.to_csv(fname, index=False)


def save_test_accuracy(
    optimizers: Dict[str, DefaultOptimizer], fname: str
) -> None:
    data = {}
    for label, optimizer in optimizers.items():
        data[label] = optimizer.test_accuracy

    with open(fname, 'w') as f:
        json.dump(data, f)
        f.close()


def get_networks() -> Dict[str, SpikingNetwork]:
    """Get a set of spiking networks to train."""
    somatic_spike_fn = get_spike_fn(threshold=15)
    dendritic_nl_fn = get_default_dendritic_fn(
        threshold=2, sensitivity=10, gain=1
    )
    neuron_params = RecurrentNeuronParameters(
        tau_mem=10e-3,
        tau_syn=5e-3,
        backprop_gain=0.5,
        feedback_strength=15,
        somatic_spike_fn=somatic_spike_fn,
        dendritic_spike_fn=dendritic_nl_fn,
    )

    # parallel_params = PRCNeuronParameters(
    #     tau_mem=10e-3,
    #     tau_syn=5e-3,
    #     backprop_gain=0.05,
    #     feedback_strength=15,
    #     somatic_spike_fn=somatic_spike_fn,
    #     dend_na_fn=dendritic_nl_fn,
    #     dend_ca_fn=get_sigmoid_fn(threshold=4, sensitivity=10, gain=1),
    #     dend_nmda_fn=dendritic_nl_fn,
    #     tau_dend_na=5e-3,
    #     tau_dend_ca=40e-3,
    #     tau_dend_nmda=80e-3,
    # )

    simple_network_architecture = deepcopy(NETWORK_ARCHITECTURE)
    simple_network_architecture.weight_scale_by_layer = (3, 7)

    two_compartment_network_architecture = deepcopy(NETWORK_ARCHITECTURE)
    two_compartment_network_architecture.weight_scale_by_layer = (0.5, 7)

    # parallel_network_architecture = deepcopy(NETWORK_ARCHITECTURE)
    # parallel_network_architecture.weight_scale_by_layer = (0.02, 7)

    nets = {
        'One compartment': SpikingNetwork(
            neuron_params, simple_network_architecture
        ),
        'No BAP': TwoCompartmentSpikingNetwork(
            neuron_params, two_compartment_network_architecture
        ),
        'BAP': RecurrentSpikingNetwork(
            neuron_params, two_compartment_network_architecture
        ),
        # 'Parallel subunits, no BAP': ParallelSpikingNetwork(
        #     parallel_params, parallel_network_architecture
        # ),
        # 'Parallel subunits + BAP (full PRC model)': PRCSpikingNetwork(
        #     parallel_params, parallel_network_architecture
        # ),
    }
    return nets


def classification_accuracy(dataloader:DataLoader, net: SpikingNetwork) -> float:
    """ Computing classification accuracy on supplied data for each of the networks. """
    accuracies = []
    for x_local, y_local in dataloader:
    
    # sparse_data_generator_from_hdf5_spikes(
    #     x_data, y_data, SWEEP_DURATION, shuffle=False,
    # ):
        accuracies.append(
            _minibatch_classification_accuracy(
                x_local, y_local, net # Removed x_loval.to_dense() as already inside the Dataset
            )
        )
    return np.mean(accuracies)


def get_optimizers(
    nets: Dict[str, SpikingNetwork]
) -> Dict[str, DefaultOptimizer]:
    return {
        key: DefaultOptimizer(
            _get_feedforward_func(nets[key]), nets[key].weights_by_layer
        )
        for key in nets
    }


def _get_feedforward_func(net):
    def feedforward(x):
        return net.run_snn(x, reset=True)[0]

    return feedforward


if __name__ == '__main__':
    main()
