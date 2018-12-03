#!/usr/bin/env python
# -*- coding : utf-8 -*-

import h5py
import os
from threading import Lock

from abc import ABCMeta, abstractmethod
import six
import json


@six.add_metaclass(ABCMeta)
class BaseDumper(object):
    """Base class for dumper classes"""  
    
    def __repr__(self):
        return '%s(%s)' %(self.__class__.__name__, self.__dict__)

    @abstractmethod
    def dump_params(self):
        """Dump the data object to the output."""
        return
    
    
class JSONDumper(BaseDumper):
    """File System Dumper class"""
    def __init__(self, filename, overwrite=False, **dumper_params):
        root, suffix = os.path.split(filename)
        if '.' not in suffix:
            self.filename = filename + '.json'
        else :
            self.filename = filename    
        self.overwrite = overwrite
        self.dumper_params = dumper_params
    
    def dump(self, model_name, **kwargs):
        """Dump the data object to the output file define by filename."""
        i = 0
        split_fn = os.path.splitext(self.filename)
        
        if not self.overwrite:
            while os.path.exists('{}_{}_{}{}'.format(split_fn[0], model_name, i, split_fn[1])):
                i += 1

        try:
            with os.fdopen(os.open('{}_{}_{}{}'.format(split_fn[0], model_name, i, split_fn[1]),
                                   os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), 'w', encoding='utf-8') as fh:
                json.dump(kwargs, fh, separators=(',', ':'), sort_keys=True, indent=4)
                
        except OSError as e:
            if e.errno == errno.EEXIST:
                print("File %s already exists!" %('{}_{}_{}{}'.format(split_fn[0], model_name, i, split_fn[1])))

        return


class HDF5Dumper(BaseDumper):
    """File System Dumper class"""
    def __init__(self, filename, overwrite=False, **dumper_params):
        self.filename = filename
        self.overwrite = overwrite
        self.dumper_params = dumper_params

    def _modify_filename(self, filename, prefix, add_num):
        """Modify the filename when user avoid the overwrite"""
        root, suffix = os.path.split(filename)
        path = '{}/{}_{}_{}'.format(root, prefix, add_num, suffix)
        return path

    def dump(self, model_name, **kwargs):
        """Dump y_[ids, true, pred] and data to the output file define by filename"""
        num_add = 1
        file_saved = self.filename
        lock = Lock()
        lock.acquire()
        
        if not self.overwrite:
            while os.path.exists(file_saved):
                file_saved = self._modify_filename(self.filename, model_name, num_add)
                num_add += 1
            print('HDF5 FILE NAME %s' %(file_saved))
        
        with h5py.File(self.filename, 'w') as h5f:
            for key, item in kwargs.items():
                try:
                    h5f.create_dataset(key, data=np.asarray(item))
                except TypeError:
                    dt = h5py.special_dtype(vlen=bytes)
                    h5f.create_dataset(key, data=item, dtype=dt)
        lock.release()
        return