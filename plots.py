import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import plotly.express as px
import os

from ptools.lipytools.stats import stats_pd, msmx


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
    if save_FD:
        if not os.path.isdir(save_FD): os.mkdir(save_FD)
        plt.savefig(f'{save_FD}/{name}.png')
    else:
        plt.show()


def two_dim(
        y :list or np.array, # two/one dim list or np.array
        x :list or np.array=    None,
        name=                   'values',
        save_FD: str =          None):

    if type(y) is list: y = np.array(y)
    if x is None:
        if len(y.shape) < 2: x = np.arange(len(y))
        else:
            x = y[:,1]
            y = y[:,0]

    plt.clf()
    plt.plot(x,y)
    plt.legend(loc='upper right')
    plt.grid(True)
    if save_FD: plt.savefig(f'{save_FD}/{name}.png')
    else:       plt.show()


def three_dim(
    xyz: list or np.array, # list of (x,y,z)
    name=               'values',
    x_name=             'x',
    y_name=             'y',
    z_name=             'z',
    save_FD: str =      None):

    df = pd.DataFrame(
        data=       xyz,
        columns=    [x_name,y_name,z_name])

    std = df[z_name].std()
    mean = df[z_name].mean()
    off = 2*std
    cr_min = mean - off
    cr_max = mean + off

    fig = px.scatter_3d(
        data_frame=     df,
        title=          name,
        x=              x_name,
        y=              y_name,
        z=              z_name,
        color=          z_name,
        range_color=    [cr_min,cr_max],
        opacity=        0.7,
        width=          700,
        height=         700)

    if save_FD:
        file = f'{save_FD}/{name}_3Dplot.html'
        fig.write_html(file, auto_open=False if os.path.isfile(file) else True)
    else: fig.show()



if __name__ == '__main__':

    # *********************** histogram example
    histogram(np.array([1, 4, 5, 5, 3, 3, 5, 65, 32, 45, 5, 5, 6, 33, 5]))


    # *********************** two_dim example
    n = 100
    x = (np.arange(n)-n//2)/(n/2/np.pi/3)
    y = np.sin(x)

    two_dim(y,x)
    two_dim(list(zip(y,x)))
    two_dim(y)


    # *********************** three_dim example
    from ptools.neuralmess.layers import positional_encoding
    width = 1#5

    pe = positional_encoding(90, width, 0.9, 7, verb=1)
    pe = np.squeeze(pe)

    print(pe.shape)
    if width == 1:
        two_dim(pe)
    else:
        two_dim(pe[:, 0]) # first
        #two_dim(pe[:, 1])
        #two_dim(pe[:,-2])
        two_dim(pe[:,-1]) # last

    if width == 1:
        xyz = []
        for rix in range(pe.shape[0]):
            xyz.append([rix, 0, pe[rix]])
    else:
        xyz = []
        for rix in range(pe.shape[0]):
            for eix in range(pe.shape[1]):
                xyz.append([rix,eix,pe[rix,eix]])

    three_dim(xyz)