"""

 2020 (c) piteren

"""

import numpy as np

from matplotlib import pyplot as plt

from lipytools.stats import stats_pd, msmx

# draws/saves histogram for list of values, print_pd option
def histogram(
        val_list :list or np.array,
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
    histogram(np.array([1,4,5,5,3,3,5,65,32,45,5,5,6,33,5]))