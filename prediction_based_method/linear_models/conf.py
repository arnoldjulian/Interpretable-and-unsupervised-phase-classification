# path for the data folders
path_data_noisefree = '../../ms_models/L20x20_all_minsorted/'
path_data_noisy = '../../ms_models/L20x20_all_sorted/'

# type of input to be used; choose between 'raw', 'ft', and 'corr_func'
input_type = 'ft'

# number of random symmetry transformations to be applied to each configuration sample for computing the average input
num_trafos = 1

# choose between the 'noise-free' and 'noisy' case
case = 'noisy'

# dimension of the parameter space to be analyzed; choose between 1 and 2
dim_parameter_space = 1

# if 'dim_parameter_space = 1', specify the value of Nf along which the system should be analyzed
nf = 100

# set range of U under investigation. can be modified: 1 < U_min, U_max < 8 and U_min, U_max % 0.2 = 0
U_min = 1.0
U_max = 8.0
U_step = 0.2

# choose between standard linear regression ('stand') or Rigde regression ('l2')
regression_type = 'stand'

# if using L2-regularization, specify the corresponding regularization rate
l2_lambda = 0.0

# set a random seed
seed = 222
