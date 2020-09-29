import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

# plot indicator along selected linescan of constant Nf


def plot_linescan(U_range, Y_true_U, Y_pred_U, indicator_array, dim):

    # construct color map from selected colors
    basic_cols = ['black', 'darkred', 'indianred', 'red']
    my_cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)

    # construct normalization for color bar and color scale
    elev_min = 0
    elev_max = np.max(indicator_array)
    mid_val = np.mean(indicator_array)
    norm = MidpointNormalize(midpoint=mid_val, vmin=elev_min, vmax=elev_max)

    # plot indicator along selected linescan
    fig, ax = plt.subplots()
    ax.scatter(Y_true_U[1:-1], indicator_array, c=indicator_array, cmap=my_cmap,
               norm=norm, label='divergence')

    # set ticks and axis labels
    ax.set_xlabel('$U$')
    ax.set_ylabel('$\partial \delta U/\partial U$')
    ax.tick_params(axis='y', which='both', labelcolor='k')

    # plot predictions \hat{U} along selected linescan
    ax2 = ax.twinx()
    ax2.plot(U_range, Y_pred_U, color='b', linestyle='dashed')

    # set ticks and axis labels
    ax2.tick_params(axis='y', which='both', colors='b')
    ax2.set_ylabel('$\hat{U}$', color='b', ha='right', labelpad=10)

    ax2.set_yticks([2, 4, 6, 8])
    ax.set_xticks([2, 4, 6, 8])
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

    # save and close plot
    plt.savefig('result_linear_models_{}_{}_nf_{:.0f}.pdf'.format(
        conf.case, conf.input_type, conf.nf_tar))
    plt.close(fig)
