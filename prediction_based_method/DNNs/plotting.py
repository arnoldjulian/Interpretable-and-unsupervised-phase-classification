import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import conf

# midpoint normalization which is applied to color scales and color bars, such that a diverging scale from a fixed midpoint value is obtained


class MidpointNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None): .
    x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
    return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


# set training epoch for which the model should be evaluated, i.e., for which a predicted phase diagram and loss curve should be plotted
epoch = 2


if conf.dim_parameter_space == 1:

    # load underlying system parameters, predictions and corresponding divergence of the model at the specified training epoch
    p0 = np.genfromtxt('../div/p0_epoch{}.txt'.format(epoch)).reshape(-1)
    ppred = np.genfromtxt('../div/ppred_epoch{}.txt'.format(epoch)).reshape(-1)
    dp = np.genfromtxt('../div/dp_epoch{}.txt'.format(epoch)).reshape(-1)
    divp = np.genfromtxt('../div/divp_epoch{}.txt'.format(epoch)).reshape(-1)

    # plot the predictions and corresponding divergence along the specified linescan and save it in the final_figures folder
    plt.plot(p0, ppred, '-r', label='p0')
    plt.plot(p0, dp, '-g', label='dp')
    plt.plot(p0, divp, '-b', label='divp')
    plt.xlabel('$U$')
    plt.ylabel('div')
    plt.legend()
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

    # draw 2D contour plot of indicator
    contplot = plt.contourf(p0[1:-1, :, 0], p0[1:-1, :, 1]/conf.dim**2, divp[1:-1, :], 100, cmap=my_cmap,
                            norm=norm)

    for c in contplot.collections:
        c.set_edgecolor("face")

    # plot the vector-field divergence in the two-dimensional parameter space and save it in the final_figures folder
    plt.quiver(p0[1:-1, 1:-1, 0], p0[1:-1, 1:-1, 1]/conf.dim**2, dp[1:-1, 1:-1, 0], dp[1:-1, 1:-1, 1]/conf.dim**2,
               units='width',
               angles='xy', scale_units='xy', scale=1.,
               color='white')
    plt.xlabel('$U$')
    plt.ylabel('$\\rho$')
    plt.draw()
    plt.savefig('../final_figures/indicator_epoch{}.pdf'.format(epoch))
    plt.close()

# plot the loss curve of the model up to the specified training epoch and save it in the final_figures folder
input = np.genfromtxt('../logs/logs.txt', skip_header=1, delimiter=', ')
epochs = input[:, 0]
loss = input[:, 1]

plt.plot(epochs, loss, '-r', label='loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.yscale('log')
plt.legend()
plt.draw()
plt.savefig('../final_figures/loss_epoch{}.pdf'.format(epoch))
plt.close()
