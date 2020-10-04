# path for the data folders
path_data_noisefree = '../../../ms_models/L20x20_all_minsorted/'
path_data_noisy = '../../../ms_models/L20x20_all_sorted/'

# type of input to be used; choose between 'raw', 'ft', and 'corr_func'
input_type = 'ft'

# number of random symmetry transformations to be applied to each configuration sample for computing the average input (will be ignored when choosing correlation functions as inputs)
num_trafos = 20

# choose between the 'noise-free' and 'noisy' case; we provide data for the noise-free case
case = 'noise-free'

# dimension of the parameter space to be analyzed; fixed to 1 for linear models
dim_parameter_space = 1

# if 'dim_parameter_space = 1', specify the value of Nf along which the system should be analyzed
nf = 63

# set range of U under investigation. can be modified: 1 < U_min, U_max < 8 and U_min, U_max % 0.2 = 0
U_min = 1.0
U_max = 8.0
U_step = 0.2

# choose between standard linear regression ('stand') or Rigde regression ('l2')
regression_type = 'l2'

# if using Rigde regression, specify the corresponding L2-regularization rate
l2_lambda = 1e-7

# set a random seed
seed = 222
