import torch
import time
import numpy as np
from pathlib import Path
import random
import conf
import dataloader

# perform lattice translations on a given input configuration


def roll(tensor, axis, shift):
    if shift == 0:
        return tensor

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, after_start)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)

# perform reflections on a given input configuration


def mirror(x, axis, shift):
    if shift == 0:
        return x

    if shift == 1:
        return torch.flip(x, [axis])
    else:
        print("something wrong in mirror")

# perform rotations on a given input configuration


def rotate(x, shift):
    if shift == 0:
        return x
    else:
        return torch.rot90(x, shift, [1, 2])

# Running stats class to calculate the mean and std in an online fashion based on Welfords algorithm from https://stackoverflow.com/questions/1174984/how-to-efficiently-calculate-a-running-standard-deviation


class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return np.sqrt(self.variance())

    def standard_deviation_mean(self):
        return self.standard_deviation() / np.sqrt(self.n)


# Timer object to check run times of indiviual steps
class Timer(object):
    def __init__(self, name=None, loc_txt_file='../logs/logs.txt'):
        self.name = name
        self.loc_txt_file = loc_txt_file

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))
        with open(self.loc_txt_file, "a") as txt_file:
            txt_file.write('{:.6f}'.format(time.time() - self.tstart) + '\n')

# function which loops over all subdirectories of the scan (data folder) and assigns them to train and test data sets. Here, we do not consider training and test sets separately, but merely a general, overall dataset.


def construct_partitions_FKM(path=''):
    if conf.case == 'noise-free':
        v_range = range(1)
    else:
        v_range = range(1, 11)

    dataset = []
    if conf.dim_parameter_space == 1:
        for U in np.arange(conf.U_min, conf.U_max+conf.U_step, conf.U_step):
            for v in v_range:
                filename = path + \
                    '/U{:.1f}/Nf{:.0f}/v{:.0f}/conf.dat'.format(U, conf.nf_tar, v)
                dataset.append(filename)

    elif conf.dim_parameter_space == 2:
        for nf in range(1, conf.num_nf+1):
            for U in np.arange(conf.U_min, conf.U_max+conf.U_step, conf.U_step):
                for v in v_range:
                    filename = path + '/U{:.1f}/Nf{:.0f}/v{:.0f}/conf.dat'.format(U, nf, v)
                    dataset.append(filename)

    else:
        print('Parameter space dimension is not supported!')

    return dataset, dataset
