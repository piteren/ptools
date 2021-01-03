"""

 2019 (c) piteren

    stats methods

"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# min, avg, max ..of num list
def mam(vals: list):
    return [min(vals), sum(vals) / len(vals), max(vals)]

# mean, std, min, max (from given list of values or np.arr)
def msmx(vals : list or np.array) -> dict:

    arr = np.asarray(vals) if type(vals) is list else vals
    ret_dict = {
        'mean': float(np.mean(arr)),
        'std':  float(np.std(arr)),
        'min':  float(np.min(arr)),
        'max':  float(np.max(arr))}
    ret_dict['string'] = 'mean %.5f, std %.5f, min %.5f, max %.5f'%(ret_dict['mean'],ret_dict['std'],ret_dict['min'],ret_dict['max'])
    return ret_dict

# deep stats (with pandas)
def stats_pd(
        val_list :list,
        n_percentiles=  10) -> str:
    return f'{pd.Series(val_list).describe(percentiles=[0.1*n for n in range(1,n_percentiles)])}'

# draws/saves histogram for list of values, print_pd option
def histogram(
        val_list :list,
        name=           'values',
        rem_nstd :int=  0,      # removes values out of N*STD
        print_pd=       True,
        density=        True,
        bins=           20,
        save_FD :str=   None):

    if print_pd:
        print(f'\nStats with pandas:')
        print(stats_pd(val_list))

    if rem_nstd:
        stats = msmx(val_list)
        std = stats['std']
        mean = stats['mean']
        n = rem_nstd
        val_list = [val for val in val_list if mean - n * std < val < mean + n * std]

        if print_pd:
            print(f'\nStats after removing {rem_nstd} STD:')
            print(stats_pd(val_list))

    plt.clf()
    plt.hist(val_list, label=name, density=density, bins=bins, alpha=0.5)
    plt.legend(loc='upper right')
    plt.grid(True)
    if save_FD: plt.savefig(f'{save_FD}/{name}.png')
    else:       plt.show()


if __name__ == '__main__':
    histogram([1,4,5,5,3,3,5,65,32,45,5,5,6,33,5])
