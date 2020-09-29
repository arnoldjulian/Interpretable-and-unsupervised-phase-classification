import numpy as np

# set a random seed
seed = 200

# dimension of parameter space under consideration
dim_parameter_space = 2

# range of U
U_min = 1.0
U_max = 8.0
U_step = 0.2

# if dim_parameter_space = 1, additionally give the particular Nf which should be considered for a linescan
nf_tar = 3

# linear lattice size L
dim = 20

# define variables which quantify the parameter space
U_scale = 1/np.std(np.arange(U_min, U_max+U_step, U_step))
num_U = int(round(((U_max-U_min)/U_step)+1))
num_nf = int(round(dim**2/2))

if dim_parameter_space == 1:
    nf_scale = 0
    hder = U_step*U_scale
    num_nf = 1

elif dim_parameter_space == 2:
    nf_scale = 1/np.std(range(1, num_nf+1))
    hder = [U_step*U_scale, 1*nf_scale]


# path to the folder containing the data
data_dir = '../L'+str(dim)+'x'+str(dim) + '_all_sorted'

# type of input to be used; choose between 'raw', 'ft', and 'corr_func'
input_type = 'ft'

# choose between the 'noise-free' and 'noisy' case
case = 'noisy'

# number of epochs to be trained
epochs = 3000

# Epochs at which a predicted phase diagram should be plotted and saved
epoch_step = 50
saving_epochs = np.arange(0, epochs+epoch_step, epoch_step)
for i in range(1, len(saving_epochs)):
    saving_epochs[i] -= 1

# specify NN hyperparameters

# batch size
batch_size = 200*num_U

# activation function
actfunc = 'relu'

# parameters for training in batch mode
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 1}
params_stand = {'batch_size': 1, 'shuffle': True, 'num_workers': 1}

# learning rate (Adam minimizer)
lr = 1e-4

# l2 regularization rate
l2_lambda = 0.0

# parameters for the learning rate scheduler (ReduceOnPlateau)
lr_scheduler_factor = 0.5
lr_scheduler_patience = 50

# NN architecture
channels = [1, 2048]
kernels = [20]
strides = [1]
fcs = [2048, 1024, 512, 512, 256, 2]
