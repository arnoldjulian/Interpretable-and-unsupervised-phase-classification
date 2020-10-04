import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import conf

# midpoint normalization which is applied to color scales and color bars, such that a diverging scale from a fixed midpoint value is obtained


class MidpointNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


# set font
font = {'family': 'serif',
        'weight': 'normal',
        'size': 24}

# construct color map from selected colors
basic_cols = ['blue', 'mediumblue', 'darkblue', 'black', 'darkred', 'indianred', 'red']
my_cmap = LinearSegmentedColormap.from_list('mycmap', basic_cols)


# cut the first and last value as there are large peaks in the divergence due to the numerical computation

# class that contains the routines to plot the results
@torch.no_grad()
class myplotter():

    def __init__(self):
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.epoch_dict = {'train': [], 'test': []}
        self.loss_dict = {'train': [], 'test': []}

    # function to keep track of (training and test) loss as a function of epochs
    def loss_data(self, epoch, loss, phase):
        self.epoch_dict[phase].append(epoch)
        self.loss_dict[phase].append(loss.cpu().numpy())

    # function to plot the divergence along a 1D horizontal linecan at Nf=const. given the model predictions
    def pltdiv1d(self, p0, ppred, dp, divp):
        U_scale_plot = 1/(conf.U_max*conf.U_scale)

        # clear axis of plot
        self.axes[1].cla()

        # put tensors to numpy for plotting
        p0_np = p0.cpu().numpy()
        p0_plt = p0_np[1:-1]*U_scale_plot
        p0_save = p0_np/conf.U_scale

        ppred_np = ppred.cpu().numpy()
        ppred_plt = ppred_np[1:-1]*U_scale_plot
        ppred_save = ppred_np/conf.U_scale

        dp_np = dp.cpu().numpy()
        dp_plt = dp_np[1:-1]*U_scale_plot
        dp_save = dp_np/conf.U_scale

        divp_np = divp.cpu().numpy()
        divp_plt = divp_np[1:-1]

        self.axes[1].plot(p0_save[1:-1], dp_plt, label='$\delta U$')
        self.axes[1].plot(p0_save[1:-1], ppred_plt, label='$\hat{U}$')
        self.axes[1].plot(p0_save[1:-1], divp_plt, label='$\mathrm{div}(\delta U)$')
        self.axes[1].set_xlabel('$U$')
        self.axes[1].legend()

        return p0_save, ppred_save, dp_save, divp_np

    # function to plot the vector field and the corresponding divergence in 2D parameter space given the model predictions
    def pltdiv2d(self, p0, ppred, dp, divp):
        U_scale_plot = 1/(conf.U_max*conf.U_scale)
        nf_scale_plot = 1/(conf.num_nf*conf.nf_scale)

        # clear axis of plot
        self.axes[1].cla()

        # put tensors to numpy for plotting
        p0_np = p0.cpu().numpy()
        p0_plt = np.zeros(p0[1:-1, 1:-1, :].shape)
        p0_save = np.zeros(p0.shape)

        p0_plt[:, :, 0] = p0_np[1:-1, 1:-1, 0]*U_scale_plot
        p0_plt[:, :, 1] = p0_np[1:-1, 1:-1, 1]*nf_scale_plot
        p0_save[:, :, 0] = p0_np[:, :, 0]/conf.U_scale
        p0_save[:, :, 1] = p0_np[:, :, 1]/conf.nf_scale

        ppred_np = ppred.cpu().numpy()
        ppred_plt = np.zeros(ppred[1:-1, 1:-1, :].shape)
        ppred_save = np.zeros(ppred.shape)

        ppred_plt[:, :, 0] = ppred_np[1:-1, 1:-1, 0]*U_scale_plot
        ppred_plt[:, :, 1] = ppred_np[1:-1, 1:-1, 1]*nf_scale_plot
        ppred_save[:, :, 0] = ppred_np[:, :, 0]/conf.U_scale
        ppred_save[:, :, 1] = ppred_np[:, :, 1]/conf.nf_scale

        dp_np = dp.cpu().numpy()
        dp_plt = np.zeros(dp[1:-1, 1:-1, :].shape)
        dp_save = np.zeros(dp.shape)

        dp_plt[:, :, 0] = dp_np[1:-1, 1:-1, 0]*U_scale_plot
        dp_plt[:, :, 1] = dp_np[1:-1, 1:-1, 1]*nf_scale_plot
        dp_save[:, :, 0] = dp_np[:, :, 0]/conf.U_scale
        dp_save[:, :, 1] = dp_np[:, :, 1]/conf.nf_scale

        divp_np = divp.cpu().numpy()
        divp_plt = divp_np[1:-1, 1:-1]

        elev_min = np.min(divp_plt)
        elev_max = np.max(divp_plt)
        mid_val = 0
        norm = MidpointNormalize(midpoint=mid_val, vmin=elev_min, vmax=elev_max)

        contplot = self.axes[1].contourf(
            p0_plt[:, :, 0], p0_plt[:, :, 1], divp_plt, 100, cmap=my_cmap, norm=norm)

        self.axes[1].quiver(p0_plt[:, :, 0], p0_plt[:, :, 1], dp_plt[:, :, 0], dp_plt[:, :, 1],
                            units='width',
                            angles='xy', scale_units='xy', scale=1.,
                            color='silver')
        self.axes[1].set_xlabel('$U/U_\mathrm{max}$')
        self.axes[1].set_ylabel('$\\rho$')

        return p0_save, ppred_save, dp_save, divp_np

    # function to assign the proper plotting routine given the chosen dimension of the investigated parameter space
    def pltdiv(self, p0, ppred, dp, divp):
        if conf.dim_parameter_space == 1:
            p0_save, ppred_save, dp_save, divp_save = self.pltdiv1d(p0, ppred, dp, divp)
        elif conf.dim_parameter_space == 2:
            p0_save, ppred_save, dp_save, divp_save = self.pltdiv2d(p0, ppred, dp, divp)
        else:
            exit('self.dim != 1 or 2')
        return p0_save, ppred_save, dp_save, divp_save

    # function to plot the training and test loss as a function of the epochs
    def _draw(self, time):

        # clear axis of plot
        self.axes[0].cla()
        self.axes[0].plot(self.epoch_dict['train'], self.loss_dict['train'])
        self.axes[0].set_xlabel('epochs')
        self.axes[0].set_ylabel('$L_\mathrm{MSE}$')
        self.axes[0].set_yscale('log')

        self.fig.tight_layout()
        plt.draw()
        plt.pause(time)

    def close(self):
        plt.close()
