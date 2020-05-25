"""

 2020 (c) piteren

"""

import numpy as np

from matplotlib import pyplot as plt

# draws/saves histogram for list of values, print_pd option
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


if __name__ == '__main__':

    n = 100
    x = (np.arange(n)-n//2)/(n/2/np.pi/3)
    y = np.sin(x)

    two_dim(y,x)
    two_dim(list(zip(y,x)))
    two_dim(y)

