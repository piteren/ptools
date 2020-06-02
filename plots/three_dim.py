"""

 2020 (c) piteren

"""

import numpy as np
import pandas as pd
import plotly.express as px
import os


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

    from ptools.neuralmess.layers import positional_encoding
    width = 1#5

    pe = positional_encoding(90, width, 0.9, 7, verb=1)
    pe = np.squeeze(pe)

    from ptools.plots.two_dim import two_dim

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