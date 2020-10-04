# path for the data folders
path_data_noisefree = '../data/'
path_data_noisy = '../data/'

# type of input to be used; choose between 'raw', 'ft', and 'corr_func'
input_type = 'ft'

# number of random symmetry transformations to be applied to each configuration sample for computing the average input (ignored when choosing correlation functions as inputs)
num_trafos = 20

# choose between the 'noise-free' and 'noisy' case; we provide data for the noise-free case
case = 'noisy'

# dimension of the parameter space to be analyzed; choose between 1 and 2
dim_parameter_space = 1

# if 'dim_parameter_space = 1', specify the value of Nf along which the system should be analyzed
nf = 63

# set range of U under investigation. can be modified: 1 < U_min, U_max < 8 and U_min, U_max % 0.2 = 0
U_min = 1.0
U_max = 8.0
U_step = 0.2

# set a random seed
seed = 222
