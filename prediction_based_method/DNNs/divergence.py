import utils
import torch
import numpy as np
import visualization
import conf


@torch.no_grad()
class vfdivergence():
    """
    class that contains the routines to compute the vector-field divergence
    """

    def __init__(self):
        # initialize dictionaries

        # p0_dict: stores the combinations of p0 that already occured
        self.p0_dict = {}

        # ppred_dict: compute average predictions on the grid
        self.ppred_dict = {}

    # reset the dictionary
    def reset_dict(self):
        self.p0_dict = {}
        self.ppred_dict = {}

    # when processing batches, push the batch of true and predicted parameters to calculate mean and std in an online fashion
    def push_data(self, p0, ppred):

        for j, x in enumerate(p0):

            current_key = tuple(x.view(-1).tolist())

            if current_key in self.p0_dict:
                self.ppred_dict[current_key].push(1.0*ppred[j])

            else:

                self.p0_dict[current_key] = 1*x

                # online mean and std
                self.ppred_dict[current_key] = utils.RunningStats()
                self.ppred_dict[current_key].push(1.0*ppred[j])

    # get averaged predictions from current online mean
    def averaging(self):
        out_ppred = []
        out_p0 = []

        for current_key in self.p0_dict:

            ppred = self.ppred_dict[current_key].mean()
            p0 = self.p0_dict[current_key]

            out_ppred.append(ppred)
            out_p0.append(p0)

        self.out_ppred = torch.stack(out_ppred)
        self.out_p0 = torch.stack(out_p0)

        p0, ppred = self.sort()

        self.reset_dict()

        return p0, ppred

    def sort(self):

        if self.out_p0.size(1) == 1:

            p0sorted, indices = torch.sort(self.out_p0, dim=0)
            return p0sorted, self.out_ppred[indices.squeeze()]

        elif self.out_p0.size(1) == 2:
            param1 = self.out_p0[:, 0].cpu().numpy()
            param2 = self.out_p0[:, 1].cpu().numpy()

            # lexsort: https://www.geeksforgeeks.org/sort-sorteda-np-argsorta-np-lexsortb-python/
            indices = np.lexsort((param2, param1))

            return self.out_p0[indices], self.out_ppred[indices]

        else:
            print('Only 1d and 2d are supported.')

    # calculate divergence in 1D using symmetric differnce quotient
    def divergence1d(self, p0, ppred):

        hder = 2*conf.hder

        dp = ppred-p0

        div = (utils.roll(dp, axis=0, shift=-1)-utils.roll(dp, axis=0, shift=1))/hder

        # reset dictionaries
        self.reset_dict()

        return p0, ppred, dp, div

    # calculate divergence in 2D using symmetric differnce quotient
    def divergence2d(self, p0, ppred):

        hder = 2*conf.hder

        p0 = p0.view(conf.num_U, conf.num_nf, 2)
        ppred = ppred.view(conf.num_U, conf.num_nf, 2)

        dp = ppred-p0
        div = (utils.roll(dp[:, :, 0], axis=0, shift=-1)-utils.roll(dp[:, :, 0], axis=0, shift=1))/hder[0] + (
            utils.roll(dp[:, :, 1], axis=1,  shift=-1)-utils.roll(dp[:, :, 1], axis=1, shift=1))/hder[1]

        # reset dictionaries
        self.reset_dict()

        return p0, ppred, dp, div
