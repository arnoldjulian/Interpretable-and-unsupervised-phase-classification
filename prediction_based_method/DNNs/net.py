import numpy as np
import torch
import torch.nn as nn
import conf

# neural network (NN)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Define dictionary for activation functions
        actfuncs = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh(), 'elu': nn.ELU()}

        # set activation function
        self.actfunc = actfuncs[conf.actfunc]

        # define neural network

        if conf.input_type == 'corr_func':
            num_inputs = 3*int(round(conf.dim/2))

            conf.fcs[0] = num_inputs

            fc_layers = []
            n_fc = len(conf.fcs)-1
            for i in range(n_fc):
                fc_layers.extend([nn.Linear(conf.fcs[i], conf.fcs[i+1]),
                                  self.actfunc]
                                 )
            fc_layers.pop()

            self.fc_layers = nn.Sequential(*fc_layers)

        else:
            # convolutional layers
            conv_layers = []

            # number of convolutions
            n_conv = len(conf.kernels)

            for i in range(n_conv):
                conv_layers.extend([nn.Conv2d(conf.channels[i], conf.channels[i+1],
                                              conf.kernels[i], conf.strides[i]), self.actfunc])

            # fully-connected (fc) layers
            fc_layers = []
            n_fc = len(conf.fcs)-1
            for i in range(n_fc):
                fc_layers.extend([nn.Linear(conf.fcs[i], conf.fcs[i+1]),
                                  self.actfunc]
                                 )
            fc_layers.pop()

            self.conv_layers = nn.Sequential(*conv_layers)
            self.fc_layers = nn.Sequential(*fc_layers)

    # computing an output 'out' given an input 'x' and the above defined neural network
    def forward(self, x):
        if conf.input_type == 'corr_func':
            x = x.view(-1, self.num_flat_features(x))
            out = self.fc_layers(x)
        else:
            out = self.conv_layers(x)
            out = out.view(-1, self.num_flat_features(out))
            out = self.fc_layers(out)

        return out

    # compute number of flat features
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension\n",
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
