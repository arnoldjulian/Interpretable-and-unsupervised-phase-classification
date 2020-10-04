import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator
import conf

# set training epoch for which the model should be evaluated, i.e., for which a predicted phase diagram and loss curve should be plotted
epoch = 5

# set rcParams to configure plots
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'axes.titlesize': 18})
plt.rcParams.update({'axes.labelsize': 18})
plt.rcParams.update({'xtick.labelsize': 18})
plt.rcParams.update({'ytick.labelsize': 18})
plt.rcParams.update({'axes.linewidth': 1.5})
plt.rcParams.update({'xtick.major.size': 6})
plt.rcParams.update({'xtick.major.width': 1.5})
plt.rcParams.update({'ytick.major.size': 6})
plt.rcParams.update({'ytick.major.width': 1.5})
plt.rcParams.update({'xtick.minor.size': 3})
plt.rcParams.update({'xtick.minor.width': 1.0})
plt.rcParams.update({'ytick.minor.size': 3})
plt.rcParams.update({'ytick.minor.width': 1.0})
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


if conf.dim_parameter_space == 1:

    # load underlying system parameters, predictions and corresponding divergence of the model at the specified training epoch
    p0 = np.genfromtxt('../div/p0_epoch{}.txt'.format(epoch)).reshape(-1)
    ppred = np.genfromtxt('../div/ppred_epoch{}.txt'.format(epoch)).reshape(-1)
    dp = np.genfromtxt('../div/dp_epoch{}.txt'.format(epoch)).reshape(-1)
    divp = np.genfromtxt('../div/divp_epoch{}.txt'.format(epoch)).reshape(-1)

    # construct color map from selected colors
    basic_cols = ['black', 'darkred', 'indianred', 'red']
    my_cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)

    # construct normalization for color bar and color scale
    elev_min = 0
    elev_max = np.max(divp[1:-1])
    mid_val = np.mean(divp[1:-1])
    norm = MidpointNormalize(midpoint=mid_val, vmin=elev_min, vmax=elev_max)

    # plot the divergence along the specified linescan
    fig, ax = plt.subplots()
    ax.scatter(p0[1:-1], divp[1:-1], c=divp[1:-1], cmap=my_cmap,
               norm=norm, label='divergence')

    # set ticks and axis labels
    ax.set_xlabel('$U$')
    ax.set_ylabel('$\partial \delta U/\partial U$')
    ax.tick_params(axis='y', which='both', labelcolor='k')

    # plot predictions \hat{U} along selected linescan
    ax2 = ax.twinx()
    ax2.plot(p0[1:-1], ppred[1:-1], color='b', linestyle='dashed')

    # set ticks and axis labels
    ax2.tick_params(axis='y', which='both', colors='b')
    ax2.set_ylabel('$\hat{U}$', color='b', ha='right', labelpad=10)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

    # save the plot in the final_figures subfolder and close it
    plt.draw()
    plt.savefig('../final_figures/indicator_epoch{}.pdf'.format(epoch))
    plt.close()

else:

    # load underlying system parameters, predictions and corresponding divergence of the model at the specified training epoch
    p0 = np.genfromtxt('../div/p0_epoch{}.txt'.format(epoch)).reshape(conf.num_U, conf.num_nf, 2)
    ppred = np.genfromtxt('../div/ppred_epoch{}.txt'.format(epoch)
                          ).reshape(conf.num_U, conf.num_nf, 2)
    dp = np.genfromtxt('../div/dp_epoch{}.txt'.format(epoch)).reshape(conf.num_U, conf.num_nf, 2)
    divp = np.genfromtxt('../div/divp_epoch{}.txt'.format(epoch)).reshape(conf.num_U, conf.num_nf)

    # construct color map from selected colors
    basic_cols = ['blue', 'mediumblue', 'darkblue', 'black', 'darkred', 'indianred', 'red']
    my_cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)

    # construct normalization for color bar and color scale
    elev_min = np.min(divp[1:-1, 1:-1])
    elev_max = np.max(divp[1:-1, 1:-1])
    mid_val = 0
    norm = MidpointNormalize(midpoint=mid_val, vmin=elev_min, vmax=elev_max)

    # draw 2D contour plot of indicator and save it in the final_figures folder
    contplot = plt.contourf(p0[1:-1, :, 0], p0[1:-1, :, 1]/conf.dim**2, divp[1:-1, :], 100, cmap=my_cmap,
                            norm=norm)

    for c in contplot.collections:
        c.set_edgecolor("face")

    plt.colorbar(cm.ScalarMappable(norm=MidpointNormalize(
        midpoint=mid_val, vmin=elev_min, vmax=elev_max), cmap=my_cmap), pad=0.02)

    plt.xlabel('$U$')
    plt.ylabel('$\\rho$')
    plt.draw()
    plt.savefig('../final_figures/indicator_epoch{}.pdf'.format(epoch))
    plt.close()

    # plot the vector-field in the two-dimensional parameter space and save it in the final_figures folder
    plt.quiver(p0[1:-1, 1:-1, 0], p0[1:-1, 1:-1, 1]/conf.dim**2, dp[1:-1, 1:-1, 0], dp[1:-1, 1:-1, 1]/conf.dim**2,
               units='width',
               angles='xy', scale_units='xy', scale=1.,
               color='blue')
    plt.xlabel('$U$')
    plt.ylabel('$\\rho$')
    plt.draw()
    plt.savefig('../final_figures/vectorfield_epoch{}.pdf'.format(epoch))
    plt.close()

# plot the loss curve of the model up to the specified training epoch and save it in the final_figures folder
input = np.genfromtxt('../logs/logs.txt', skip_header=1, delimiter=', ')
epochs = input[:, 0]
loss = input[:, 1]

plt.plot(epochs, loss, '-r')
plt.xlabel('epochs')
plt.ylabel('$\mathcal{L}_{\mathrm{MSE}}$')
plt.yscale('log')
plt.draw()
plt.savefig('../final_figures/loss_epoch{}.pdf'.format(epoch))
plt.close()
