#!/usr/bin/env python
# -*- coding : utf-8 -*-

import os
import sys 
import numpy as np
from re import sub, findall
import tensorflow as tf


__all__ = ['set_session', 'soft_gpu_allocation']


def set_session(gpu_number=1, gpu_fraction=.5, allow_growth=False, **kwargs):

    '''
    Limit the GPU memory and allocate the half of GPU ressource
    in case you're using keras with Tensoflow backend.
    
    Parameter
    ---------
    gpu_number : int, list, tuple or str with desired devices, default=1
                 Number of GPU to use

    gpu_fraction : float includes in [0,1], default=0.5
                   Fraction of GPU memory allocated to the session 

    log_device_placement : boolean,
                           To find out which devices your operations and tensors are assigned to.
    
    allow_soft_placement : boolean,
                           To automatically choose an existing and supported device to run the 
                           operations in case the specified one doesn't exist
    
    allow_growth : boolean, default=False
                   Allow the memory usage growth as is needed by the process.
                   It attempts to allocate only as much GPU memory based on runtime allocations: 
                   it starts out allocating very little memory, and as Sessions get run and more GPU
                   memory is needed, we extend the GPU memory region needed by the TensorFlow process.
                   
    Return
    ------
    Tensorflow Session
    '''

    devices = sub(r'\[*\]*\(*\)*\ *', r'', str(gpu_number))
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    num_threads = os.environ.get('OMP_NUM_THREADS')
    config = tf.ConfigProto(**kwargs)
    
    if allow_growth==True:
        config.gpu_options.allow_growth = True
    else : 
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction

    if num_threads:
        config.intra_op_parallelism_threadsi = num_threads

    return tf.Session(config=config)


def soft_gpu_allocation(qty=1, gpu_fraction=None, allow_growth=True, **kwargs):
    """
    Scan every available gpu and allocate the gpu with the biggest free mermory.
    
    Parameters
    ----------
    qty : int, default=1
          Number of GPU needed
    
    gpu_fraction : float includes in [0,1], default=None
                   Fraction of GPU memory allocated to the session 

    allow_growth : boolean, default=False
                   Allow the memory usage growth as is needed by the process.
                   It attempts to allocate only as much GPU memory based on runtime allocations: 
                   it starts out allocating very little memory, and as Sessions get run and more GPU
                   memory is needed, we extend the GPU memory region needed by the TensorFlow process.
    
    Returns
    -------
    Tensorflow session


    Example
    -------
    from limit_gpu_mem_env import soft_gpu_allocation
    from keras.backend.tensorflow_backend import set_session

    set_session(soft_gpu_allocation(qty=2))
    """
    
    # Synchronize nvidia-smi gpu_id with tensorflow device_id
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    
    # Query nvidia-smi
    f = os.popen('nvidia-smi --query-gpu=index,memory.free --format=csv,noheader')
    output = f.readlines()

    # Get gpus id ordering by memory free.
    free_mem = {k:v for (k,v) in [findall(r'\d+', t) for t in output]}
    id_gpu = np.argsort([int(i) for i in free_mem.values()])[::-1]
    gpu = list(id_gpu[:qty]) 
    
    TFsession = set_session(gpu_number=gpu, 
                            gpu_fraction=gpu_fraction,
                            allow_growth=allow_growth, 
                            **kwargs)
    return TFsession


if __name__=='__main__':

    gpu_number=[1,2] # ou a faire via sys.argv[1]?
    gpu_fraction=0.1
    s = KTF.set_session(set_session(gpu_number=gpu_number, gpu_fraction=gpu_fraction))

    from tensorflow.python.client import device_lib

    print('DEVICES : %i (including CPU and GPU)\n' %(len(device_lib.list_local_devices())),
          device_lib.list_local_devices())
    print('\nFraction of GPU memory in use : %d' %(gpu_fraction))
