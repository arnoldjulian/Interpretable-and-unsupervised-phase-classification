import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator
import conf

# set rcParams to configure plots
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'ytick.minor.visible': False})
plt.rcParams.update({'xtick.minor.visible': False})
plt.rcParams.update({'xtick.top': False})
plt.rcParams.update({'ytick.right': False})
plt.rcParams.update({'xtick.direction': 'out'})
plt.rcParams.update({'ytick.direction': 'out'})
plt.rcParams.update({'mathtext.fontset': 'cm'})
plt.rcParams["font.family"] = "Times New Roman"

# midpoint normalization which is applied to color scales and color bars, such that a diverging scale from a fixed midpoint value is obtained


class MidpointNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None): .
    x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
    return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

# plot inferred phase diagram in two-dimensional parameter space


def plot_phase_diagram(p0_array, indicator_array, dim):

    # construct color map from selected colors
    basic_cols = ['blue', 'mediumblue', 'darkblue',
                  'black', 'darkred', 'indianred', 'red']
    my_cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)

    # construct normalization for color bar and color scale
    elev_min = np.min(indicator_array)
    elev_max = np.max(indicator_array)
    mid_val = np.mean(indicator_array)
    norm = MidpointNormalize(midpoint=mid_val, vmin=elev_min, vmax=elev_max)

    # draw 2D contour plot of indicator
    fig, ax = plt.subplots()
    contplot = ax.contourf(p0_array[:, :, 0], p0_array[:, :, 1]/dim **
                           2, indicator_array, 800, cmap=my_cmap, norm=norm)
    for c in contplot.collections:
        c.set_edgecolor("face")

    # set ticks and axis labels
    ax.set_xticks([2, 4, 6])
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5])
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('$U$')
    ax.set_ylabel('$\\rho$')

    # draw colorbar
    plt.colorbar(cm.ScalarMappable(norm=MidpointNormalize(
        midpoint=mid_val, vmin=elev_min, vmax=elev_max), cmap=my_cmap), ax=ax, pad=0.02)

    # save and close plot
    plt.savefig('./result_mean_based_method_{}_{}.pdf'.format(conf.case,
                                                              conf.input_type), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# plot indicator along selected linescan of constant Nf


def plot_linescan(p0_array, indicator_array, dim):

    # construct color map from selected colors
    basic_cols = ['black', 'darkred', 'indianred', 'red']
    my_cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)

    # construct normalization for color bar and color scale
    elev_min = 0
    elev_max = np.max(indicator_array[1:-1])
    mid_val = np.mean(indicator_array[1:-1])
    norm = MidpointNormalize(
        midpoint=mid_val, vmin=elev_min, vmax=elev_max)

    # plot indicator along selected linescan
    fig, ax = plt.subplots()
    ax.scatter(p0_array[1:-1], indicator_array[1:-1],
               c=indicator_array[1:-1], cmap=my_cmap, norm=norm)

    # set ticks and axis labels
    ax.set_xticks([2, 4, 6, 8])
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('$U$')
    ax.set_ylabel('$\Delta\bar{x} -\langle \Delta\bar{x} \rangle$')

    # save and close plot
    plt.savefig('./result_mean_based_method_{}_{}_nf_{:.0f}.pdf'.format(conf.case,
                                                                        conf.input_type, conf.nf_tar), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
